import argparse
import os
import time
# from multiprocess import Pool as Pool2
from multiprocessing import Pool

import shutil
import sys
import codecs

# from onmt.utils.logging import init_logger, logger
from others import pyrouge


def process(data):
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = "rouge-tmp-{}-{}".format(current_time,pool_id)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict




def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n): # in n steps
        yield l[i:i + n]

def test_rouge(cand, ref,num_processes):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    # list of summary sentences
    candidates = [line.strip() for line in cand]
    references = [line.strip() for line in ref]

    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)
    # split sentences based on the number of processes/GPUs available
    candidates_chunks = list(chunks(candidates, int(len(candidates)/num_processes))) # n-sized sentence chunks
    references_chunks = list(chunks(references, int(len(references)/num_processes)))
    n_pool = len(candidates_chunks)
    arg_lst = []
    for i in range(n_pool):
        arg_lst.append((candidates_chunks[i],references_chunks[i],i)) # append chunks
    pool = Pool(n_pool) # number of worker processes to use = len(candidate_chunks)
    # map chunks in arg_lst to the process function to get rouge results, returns a list of results dict
    results = pool.map(process, arg_lst) # [{...}, {...}, ..., {...}]
    final_results = {}
    for i,r in enumerate(results): # r: result_dicts
        for k in r: # k: keys
            if(k not in final_results):
                final_results[k] = r[k]*len(candidates_chunks[i]) # multiply results e.g. Rouge by length of chunks
            else: # add existing rouge scores with other chunks, different rouge scores for different chunks
                final_results[k] += r[k] * len(candidates_chunks[i]) # should usually add up to candidate length
    for k in final_results: # transform results to be divided by candidate length to get the more complete result
        final_results[k] = final_results[k]/len(candidates)
    return final_results
def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
    results_dict["rouge_1_recall"] * 100,
    results_dict["rouge_2_recall"] * 100,
    # results_dict["rouge_3_f_score"] * 100,
    results_dict["rouge_l_recall"] * 100

    # ,results_dict["rouge_su*_f_score"] * 100
    )


if __name__ == "__main__":
    # init_logger('test_rouge.log')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default="candidate.txt",
                        help='candidate file')
    parser.add_argument('-r', type=str, default="reference.txt",
                        help='reference file')
    parser.add_argument('-p', type=int, default=1,
                        help='number of processes')
    args = parser.parse_args()
    print(args.c)
    print(args.r)
    print(args.p)
    if args.c.upper() == "STDIN":
        candidates = sys.stdin
    else:
        candidates = codecs.open(args.c, encoding="utf-8")
    references = codecs.open(args.r, encoding="utf-8")

    results_dict = test_rouge(candidates, references,args.p)
    # return 0
    print(time.strftime('%H:%M:%S', time.localtime())
)
    print(rouge_results_to_str(results_dict))
    # logger.info(rouge_results_to_str(results_dict))