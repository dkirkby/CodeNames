#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import warnings
import logging
import random
import gzip
import os.path

CORPUS_DIRECTORY='corpus'


def main():
    parser = argparse.ArgumentParser(
        description='Merge training corpus.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default='words.txt',
                        help='Name of word list to use.')
    parser.add_argument('-o', '--output', type=str, default='word2vec.dat',
                        help='File name for saved model.')
    parser.add_argument('--npass', type=int, default=1,
                        help='Perform this pass number.')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs to run.')
    parser.add_argument('--improve', action='store_true',
                        help='Continue to improve learning.')
    parser.add_argument('--dimension', type=int, default=300,
                        help='Dimension of word vectors to learn.')
    parser.add_argument('--min-count', type=int, default=150,
                        help='Ignore words with fewer occurences.')
    parser.add_argument('--max-distance', type=int, default=10,
                        help='Max distance between words within a sentence')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers to distribute workload across.')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=('CRITICAL', 'ERROR', 'WARNING',
                                 'INFO', 'DEBUG'),
                        help='Filter out log messages below this level.')
    args = parser.parse_args()

    # Look for an existing corpus for this pass.
    corpus_name = 'corpus_{0}.gz'.format(args.npass)
    if os.path.exists(corpus_name):

        print('Using corpus {0}'.format(corpus_name))

    else:

        # Read the wordlist into memory.
        with open(args.input, 'r') as f:
            wordlist = [w.strip().capitalize() for w in f]
        print('Read {0} words from {1}.'.format(len(wordlist), args.input))

        # Open the output corpus file for this pass.
        f_out = gzip.open(corpus_name, 'wb')

        # Perform a reproducible random shuffle of the wordlist.
        print('Shuffling the corpus for pass {0} into {1}...'
              .format(args.npass, corpus_name))
        random.seed(args.npass)
        random.shuffle(wordlist)

        # Split the wordlist into random pairs.
        for i in range(0, len(wordlist), 2):
            sentences = []
            # Read content for the first word of this pair into memory.
            in_name = os.path.join(
                CORPUS_DIRECTORY, '{0}.pre.gz'.format(wordlist[i]))
            with gzip.open(in_name, 'rb') as f_in:
                for line in f_in:
                    sentences.append(line)
            # The last "pair" might be a single.
            if i < len(wordlist) - 1:
                in_name = os.path.join(
                    CORPUS_DIRECTORY, '{0}.pre.gz'.format(wordlist[i+1]))
                # Read content for the second word of this pair into memory.
                with gzip.open(in_name, 'rb') as f_in:
                    for line in f_in:
                        sentences.append(line)

            # Shuffle sentences for this pair of words into a random order.
            sentence_order = range(len(sentences))
            random.shuffle(sentence_order)

            # Save shuffled sentences to the output corpus file.
            for j in sentence_order:
                f_out.write(sentences[j])

            print('Added {0} sentences for ({1}, {2}).'.format(
                len(sentences), wordlist[i], wordlist[i+1]))

        f_out.close()

    # Import gensim here so we can mute a UserWarning about the Pattern
    # library not being installed.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        import gensim.models.word2vec

    # Configure logging so we can monitor the learning progress.
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=getattr(logging, args.log_level))

    # Use the training sentences for this pass.
    sentences = gensim.models.word2vec.LineSentence(corpus_name)

    if args.npass > 1:
        # Load a previously trained model.
        prev_name = '{0}.{1}'.format(args.output, args.npass - 1)
        model = gensim.models.word2vec.Word2Vec.load(prev_name)
        # Update parameters from the command line.
        model.workers = args.workers
        model.iter = args.num_epochs
        # Continue training.
        model.train(sentences)
    else:
        # Train a new model.
        model = gensim.models.word2vec.Word2Vec(
            sentences, size=args.dimension, window=args.max_distance,
            min_count=args.min_count, workers=args.workers,
            sg=1, hs=1, iter=args.num_epochs)

    # Save the updated model after this pass.
    save_name = '{0}.{1}'.format(args.output, args.npass)
    model.save(save_name)


if __name__ == '__main__':
    main()
