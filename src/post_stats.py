import argparse
from os import path
from functools import reduce
import re

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def n_grams(tokens, n):
    """Get a list of n-grams(tuple) from a list of tokens"""
    l = len(tokens)
    return [tuple(tokens[i:i + n]) for i in range(l) if i + n < l]

def has_repeat(elements):
    d = set(elements) # remove repeats when initialized with sets
    return len(d) < len(elements) # bool condition True when repeats are removed, leaving len(d) smaller

def cal_self_repeat(summary):
    """Count number of n-grams repeats"""
    ngram_repeats = {2: 0, 4: 0, 8: 0}
    sents = summary.split('<q>') # split summary into a list of sentences
    for n in ngram_repeats.keys(): # bigram, 4-gram or 8-gram
        # Respect sentence boundary, returns (((n_grams_sent1 + n_grams_sent2) + n_grams_sent3) + n_grams_sent4) + ...
        # grams = list of n_gram tuples from sent1 to final sent, [tuple, tuple, ..., tuple]
        grams = reduce(lambda x, y: x + y, [n_grams(sent.split(), n) for sent in sents], []) # initializer = [](empty)
        ngram_repeats[n] += has_repeat(grams) # count number of repeats for each gram
    return ngram_repeats

def cal_novel(summary, gold, source, summary_ngram_novel, gold_ngram_novel):
    """Calculate number of novel n-grams"""
    summary = summary.replace('<q>',' ')
    summary = re.sub(r' +', ' ', summary).strip()
    gold = gold.replace('<q>',' ')
    gold = re.sub(r' +', ' ', gold).strip()
    source = source.replace(' ##','')
    source = source.replace('[CLS]',' ').replace('[SEP]',' ').replace('[PAD]',' ')
    source = re.sub(r' +', ' ', source).strip()


    for n in summary_ngram_novel.keys(): # n is the number of word grams
        summary_grams = set(n_grams(summary.split(), n)) # get set of n-grams for all three documents
        gold_grams = set(n_grams(gold.split(), n))
        source_grams = set(n_grams(source.split(), n))
        joint = summary_grams.intersection(source_grams) # get intersection between summary and source
        novel = summary_grams - joint # any other n-grams left are classed as novel
        summary_ngram_novel[n][0] += 1.0*len(novel) # tally number of novel n-grams
        summary_ngram_novel[n][1] += len(summary_grams) # tally number of n-grams in summary
        summary_ngram_novel[n][2] += 1.0 * len(novel) / (len(summary.split()) + 1e-6) # tally the proportion of novel n-grams
        joint = gold_grams.intersection(source_grams)
        novel = gold_grams - joint
        gold_ngram_novel[n][0] += 1.0*len(novel)
        gold_ngram_novel[n][1] += len(gold_grams)
        gold_ngram_novel[n][2] += 1.0 * len(novel) / (len(gold.split()) + 1e-6)


def cal_repeat(args):
    candidate_lines = open(args.result_path+'.candidate').read().strip().split('\n')
    gold_lines = open(args.result_path+'.gold').read().strip().split('\n')
    src_lines = open(args.result_path+'.raw_src').read().strip().split('\n')
    lines = zip(candidate_lines,gold_lines,src_lines)

    summary_ngram_novel = {1: [0, 0, 0], 2: [0, 0, 0], 4: [0, 0, 0]}
    gold_ngram_novel = {1: [0, 0, 0], 2: [0, 0, 0], 4: [0, 0, 0]}

    for c,g,s in lines:
        # self_repeats = cal_self_repeat(c)
        cal_novel(c, g, s,summary_ngram_novel, gold_ngram_novel) # calculate novel n-grams
    print(summary_ngram_novel, gold_ngram_novel)

    for n in summary_ngram_novel.keys():
        # summary_ngram_novel[n] = summary_ngram_novel[n][2]/len(src_lines)
        # gold_ngram_novel[n] = gold_ngram_novel[n][2]/len(src_lines)
        # calculate proportion of novel n-grams in summary
        summary_ngram_novel[n] = summary_ngram_novel[n][0]/summary_ngram_novel[n][1]
        gold_ngram_novel[n] = gold_ngram_novel[n][0]/gold_ngram_novel[n][1]
    print(summary_ngram_novel, gold_ngram_novel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='', type=str)
    parser.add_argument("-result_path", default='../../results/cnndm.0')


    args = parser.parse_args()
    eval(args.mode + '(args)')
