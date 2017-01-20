#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division

import argparse
import glob
import os

import model
from config import config


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate word embedding.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--npass', type=int,
        help='Evaluate this pass number (if not set: evaluate {0} if it is '
                        'present, or the last {0}.N in numerical order if not)'
                        .format(config.embedding))
    parser.add_argument('--top-singles', type=int, default=10,
                        help='Show top single matches.')
    parser.add_argument('--top-pairs', type=int, default=0,
                        help='Show top pair matches.')
    parser.add_argument('--save-plots', type=str, default=None,
                        help='Save plots using this filename root.')
    args = parser.parse_args()

    if args.save_plots:
        # Only import if needed and use background plotting.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    if args.npass is not None:
        evaluated_file = '{0}.{1}'.format(config.embedding, args.npass)
    elif os.path.isfile(config.embedding):
        evaluated_file = config.embedding
    else:
        all_suffixes = [f.split('.')[-1]
                        for f in glob.glob('{0}.*'.format(config.embedding))]
        evaluated_file = '{0}.{1}'.format(config.embedding,
                                          sorted(all_suffixes)[-1])

    if not os.path.isfile(evaluated_file):
        print('Embedding file {0} not found.'.format(evaluated_file))

    embedding = model.WordEmbedding(evaluated_file)

    with open(config.word_list, 'r') as f:
        words = [w.strip().lower().replace(' ', '_') for w in f]

    if args.top_singles > 0:
        best_score, saved_clues = [], []
        for word in words:
            clue, score = embedding.get_clue([word], [word], [], [])
            if clue:
                best_score.append(score)
                saved_clues.append((word, clue))
        num_clues = len(saved_clues)
        order = sorted(
            xrange(num_clues), key=lambda k: best_score[k], reverse=True)
        for i in order[:args.top_singles]:
            word, clue = saved_clues[i]
            print('{0:.3f} {1} = {2}'.format(
                best_score[i], word.upper(), clue))
        if args.save_plots:
            plt.hist(best_score, range=(0., 1.), bins=50)
            plt.xlim(0., 1.)
            plt.xlabel('Similarity Score')
            plt.ylabel('Singles')
            plt.yscale('log')
            plt.grid()
            plt.savefig(args.save_plots + '_singles.png')
            plt.clf()

    if args.top_pairs > 0:
        best_score, saved_clues = [], []
        for i1, word1 in enumerate(words):
            for i2, word2 in enumerate(words[:i1]):
                clue, score = embedding.get_clue(
                    [word1, word2], [word1, word2], [], [])
                if clue:
                    best_score.append(score)
                    saved_clues.append(((i1, i2), clue))
        num_clues = len(saved_clues)
        order = sorted(
            xrange(num_clues), key=lambda k: best_score[k], reverse=True)
        for i in order[:args.top_pairs]:
            i1, i2 = saved_clues[i][0]
            clue = saved_clues[i][1]
            print('{0:.3f} {1} + {2} = {3}'.format(
                best_score[i], words[i1].upper(), words[i2].upper(), clue))
        if args.save_plots:
            plt.hist(best_score, range=(0., 1.), bins=50)
            plt.xlim(0., 1.)
            plt.xlabel('Similarity Score')
            plt.ylabel('Pairs')
            plt.yscale('log')
            plt.grid()
            plt.savefig(args.save_plots + '_pairs.png')


if __name__ == '__main__':
    main()
