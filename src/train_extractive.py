#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import ExtSummarizer
from models.trainer_ext import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


def train_multi_ext(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size # number of GPUs = number of processes
    # use spawn context
    # The parent process starts a fresh python interpreter process. The child process will only inherit those resources
    # necessary to run the process objects run() method. In particular, unnecessary file descriptors and handles
    # from the parent process will not be inherited
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes, queues are used for message passing and communication.
    error_queue = mp.SimpleQueue() # consisting tuples of gpu rank and trace error message
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        # Initiate process for training, target: method to be run, args: arguments of the method
        # daemon: When a process exits, it attempts to terminate all of its daemonic child processes.
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        # the method (training process) blocks until the other training process whose join() method is called terminates
        p.join()


def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        # Initialize distributed training
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_single_ext(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        # allows defining custom handlers to be executed when a signal is received
        signal.signal(signal.SIGUSR1, self.signal_handler) # SIGUSR1: User-defined signal 1

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get() # return an item from the queue
        self.error_queue.put((rank, original_trace)) # put an item into the queue
        os.kill(os.getpid(), signal.SIGUSR1) # kill the current process when signal is received

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def validate_ext(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt'))) # checkpoint files
        cp_files.sort(key=os.path.getmtime) # sort files according to time of modification
        xent_lst = [] # list of losses
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst)) # get the index of the minimum xent
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3] # sort by the value: xent[0] and get the lowest three xent
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test_ext(args, device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1] # get latest modified file
                time_of_cp = os.path.getmtime(cp) # get time of modification
                if (not os.path.getsize(cp) > 0): # if the path size in bytes is smaller than 0
                    time.sleep(60) # suspends execution for 60 seconds
                    continue
                if (time_of_cp > timestep): # if modification time is larger than current timestep
                    timestep = time_of_cp # update timestep
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_ext(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args, device_id, pt, step):
    """Validate model"""
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    # called once for each serialized storage with two arguments: storage and location
    # Each serialized storage has a location tag associated with it which identifies the device it was saved from
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt']) # create dict for the checkpoint's opt attribute
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k]) # set model_flag attributes if k exists in opt
    print(args)

    model = ExtSummarizer(args, device, checkpoint) # initiate extractive summarizer
    model.eval() # set the model in evaluation mode

    # initiate data loader for the validation dataset
    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)
    #
    trainer = build_trainer(args, device_id, model, None) # build trainer class
    stats = trainer.validate(valid_iter, step) # validate model and update stats
    return stats.xent() # return loss


def test_ext(args, device_id, pt, step):
    """Test model"""
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = ExtSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter, step)

def train_ext(args, device_id):
    if (args.world_size > 1):
        train_multi_ext(args)
    else:
        train_single_ext(args, device_id)


def train_single_ext(args, device_id):
    """Train on single device"""
    init_logger(args.log_file)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed) # set seeds
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id) # set cuda device
        torch.cuda.manual_seed(args.seed) # set seed for cuda

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True # make computations deterministic (obtain the same output)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False)

    model = ExtSummarizer(args, device, checkpoint)
    optim = model_builder.build_optim(args, model, checkpoint) # build optimizer

    logger.info(model)

    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_fct, args.train_steps)
