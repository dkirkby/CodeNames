#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import io
import glob
import os.path
import re
import random
import gzip

import nltk.tokenize


def main():
    parser = argparse.ArgumentParser(
        description='Merge training corpus.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', type=str, default='corpus.txt.gz',
                        help='Name of merged corpus to write.')
    parser.add_argument('--encoding', type=str, default='utf8',
                        help='Encoding for reading corpus text.')
    parser.add_argument('--seed', type=int, default=2407,
                        help='Random seed for shuffling sentences.')
    args = parser.parse_args()

    heading = re.compile('=+ [^=]+ =+\s*')
    punctuation = (',', ';', ':', '.', '!', '?', '(', ')', '[', ']', '``', "''")

    all_sentences = []
    last_count = 0

    for topic_file in glob.glob(os.path.join('corpus', '*.txt')):
        topic_name = topic_file[7:-4]
        with io.open(topic_file, 'r', encoding=args.encoding) as f:

            # Read the whole file into memory.
            content = f.read()
            # Convert to pure ascii, ignoring any non-ascii characters.
            content = content.encode('ascii', 'ignore')
            # Remove markup headings.
            content = re.sub(heading, '', content)
            # Loop over sentences.
            for sentence in nltk.tokenize.sent_tokenize(content):
                words = []
                for token in nltk.tokenize.word_tokenize(sentence):
                    # Ignore punctuation.
                    if token in punctuation:
                        continue
                    words.append(token.lower())
                all_sentences.append(' '.join(words))

        print('{0:5d} {1}'.format(len(all_sentences) - last_count, topic_name))
        last_count = len(all_sentences)

    # Pick a random permutation of the sentences.
    random.seed(args.seed)
    order = random.sample(xrange(last_count), last_count)

    # Save the shuffled sentences to a text file.
    with gzip.open(args.output, 'wb') as f:
        for i in order:
            f.write(all_sentences[i] + '\n')

    print('Saved {0} sentences to {1}.'.format(last_count, args.output))

if __name__ == '__main__':
    main()
