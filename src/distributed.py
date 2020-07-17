""" Pytorch Distributed utils
    This piece of code was heavily inspired by the equivalent of Fairseq-py
    https://github.com/pytorch/fairseq
"""


from __future__ import print_function

import math
import pickle

import torch.distributed

from others.logging import logger


def is_master(gpu_ranks, device_id):
    return gpu_ranks[device_id] == 0


def multi_init(device_id, world_size, gpu_ranks):
    """Initiate distributed training."""

    print(gpu_ranks)
    dist_init_method = 'tcp://localhost:10000' # URL specifying how to initialize the process group
    dist_world_size = world_size # Number of processes participating in the job
    torch.distributed.init_process_group(
        backend='nccl', init_method=dist_init_method,
        world_size=dist_world_size, rank=gpu_ranks[device_id]) # rank: rank of the current process
    gpu_rank = torch.distributed.get_rank() # Rank is a unique identifier assigned to each process within a distributed process group
    if not is_master(gpu_ranks, device_id):
    #     print('not master')
        logger.disabled = True

    return gpu_rank



def all_reduce_and_rescale_tensors(tensors, rescale_denom,
                                   buffer_size=10485760):
    """All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    # return a new tensor of the same data type with size = smallest integral larger than the required bytes
    buffer_t = tensors[0].new(
        math.ceil(buffer_size / tensors[0].element_size())).zero_()
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel() # number of elements in t
            buffer_t[offset:offset+numel].copy_(t.view(-1))
            offset += numel

        # all-reduce and rescale
        torch.distributed.all_reduce(buffer_t[:offset]) # reduce tensors up to offset
        buffer_t.div_(rescale_denom) # rescale

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset:offset+numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            torch.distributed.all_reduce(t)
            t.div_(rescale_denom)
        elif filled + sz > buffer_size:
            # buffer is full, all-reduce and replace buffer with grad
            all_reduce_buffer()
            buffer = [t] # reset buffer with current t
            filled = sz # reset filled bytes to current t byte size
        else:
            # add tensor to buffer until the two conditions above activate
            buffer.append(t)
            filled += sz

    # if there are still tensors in buffer
    if len(buffer) > 0:
        all_reduce_buffer()


def all_gather_list(data, max_size=4096):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size() # number of processes in the current process group
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size)
            for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc) # length of encoding
    if enc_size + 2 > max_size:
        raise ValueError(
            'encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2:enc_size+2] = torch.ByteTensor(list(enc))

    # Gathers tensors from the whole group in a list, from in_buffer to out_buffers
    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    results = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2:size+2].tolist()) # convert ByteTensors to bytes for pickling
        result = pickle.loads(bytes_list)
        results.append(result)
    return results
