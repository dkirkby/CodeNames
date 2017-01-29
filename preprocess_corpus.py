#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import gzip
import os.path
import re

import nltk.tokenize

from config import config


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess training corpus.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', type=str, default='freqs.dat',
                        help='Filename for saving word list frequencies.')
    args = parser.parse_args()

    heading = re.compile('=+ ([^=]+) =+\s*')
    punctuation = (',', ';', ':', '.', '!', '?', '-', '%', '&', '$',
                   '(', ')', '[', ']', '{', '}', '``', "''")

    # Read the word list and find any compound words since they must be
    # treated as a single word during the learning step.
    word_list = []
    compound = {}
    total_freq, cross_freq, corpus_stats = {}, {}, {}
    with open(config.word_list, 'r') as f:
        for word in f:
            word_list.append(word.strip().capitalize())
            word = word.strip().lower()
            if ' ' in word:
                compound[word] = word.replace(' ', '_')
                freq_key = compound[word]
            else:
                freq_key = word
            # Initialize frequency counters.
            total_freq[freq_key] = cross_freq[freq_key] = 0
            corpus_stats[freq_key] = (0, 0)
    print('Wordlist contains {0} compound words:'.format(len(compound)))
    print(compound.keys())

    for word in word_list:

        freq_key = word.lower().replace(' ', '_')

        in_name = os.path.join(config.corpus_directory, config.template['articles'].format(word))
        if not os.path.exists(in_name):
            print('Skipping missing file {0}'.format(in_name))
            continue

        out_name = os.path.join(config.corpus_directory, config.template['preprocess'].format(word))
        num_sentences, num_words = 0, 0

        with gzip.open(in_name, 'rb') as f_in:
            # Read the whole file into memory.
            content = f_in.read().decode(config.encoding)
            # Remove headings.
            content = re.sub(heading, '', content)

            with gzip.open(out_name, 'wb') as f_out:
                # Loop over sentences.
                for sentence in nltk.tokenize.sent_tokenize(content):
                    words = []
                    for token in nltk.tokenize.word_tokenize(sentence):
                        # Ignore punctuation.
                        if token in punctuation:
                            continue
                        words.append(token.lower())
                    line = ' '.join(words)
                    # Replace ' ' with '_' in compound words.
                    for w in compound:
                        line = line.replace(w, compound[w])
                    # Update wordlist frequencies.
                    for w in line.split():
                        num_words += 1
                        if w in total_freq:
                            total_freq[w] += 1
                            if w != freq_key:
                                cross_freq[w] += 1
                    num_sentences += 1
                    # Save this sentence to the preprocessed output.
                    f_out.write(line.encode(config.encoding) + '\n')

        print(word, num_sentences, num_words)
        corpus_stats[freq_key] = (num_sentences, num_words)

    # Save wordlist frequencies in decreasing order.
    with open(args.output, 'w') as f_out:
        print('WORD         TOTFREQ    XFREQ    NSENT    NWORD', file=f_out)
        for w in sorted(total_freq, key=total_freq.get, reverse=True):
            print('{0:11s} {1:8d} {2:8d} {3:8d} {4:8d}'.format(
                w, total_freq[w], cross_freq[w], *corpus_stats[w]), file=f_out)
    print('Saved wordlist frequencies to {0}'.format(args.output))


if __name__ == '__main__':
    main()
