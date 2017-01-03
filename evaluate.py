#!/usr/bin/env python
from __future__ import print_function, division

import argparse

import model


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate word embedding.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default='word2vec.dat',
                        help='Name of saved model to read.')
    parser.add_argument('--wordlist', type=str, default='words.txt',
                        help='Name of word list to use.')
    parser.add_argument('--top-singles', type=int, default=10,
                        help='Show top single matches.')
    args = parser.parse_args()

    embedding = model.WordEmbedding(args.input)

    with open(args.wordlist, 'r') as f:
        words = [w.strip().lower().replace(' ', '_') for w in f]

    if args.top_singles > 0:
        best_score, saved_clues = [], []
        for word in words:
            clues = embedding.get_clues((word), (word))
            if clues:
                best_score.append(clues[0][1])
                saved_clues.append((word, clues))
        num_clues = len(saved_clues)
        order = sorted(
            xrange(num_clues), key=lambda k: best_score[k], reverse=True)
        for i in order[:args.top_singles]:
            word = saved_clues[i][0]
            clues = [w for w,s in saved_clues[i][1]]
            print('{0:.3f} {1} = {2}'.format(
                best_score[i], word.upper(), ', '.join(clues)))


if __name__ == '__main__':
    main()
