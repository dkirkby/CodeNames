#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import warnings
import logging


def main():
    parser = argparse.ArgumentParser(
        description='Merge training corpus.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default='corpus.txt.gz',
                        help='Name of merged corpus to read.')
    parser.add_argument('-o', '--output', type=str, default='word2vec.dat',
                        help='File name for saved model.')
    parser.add_argument('-n', '--num-epochs', type=int, default=10,
                        help='Number of training epochs to run.')
    parser.add_argument('--improve', action='store_true',
                        help='Continue to improve learning.')
    parser.add_argument('--dimension', type=int, default=300,
                        help='Dimension of word vectors to learn.')
    parser.add_argument('--min-count', type=int, default=45,
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

    # Import gensim here so we can mute a UserWarning about the Pattern
    # library not being installed.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        import gensim.models.word2vec

    # Configure logging so we can monitor the learning progress.
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=getattr(logging, args.log_level))

    # Read the training sentences.
    sentences = gensim.models.word2vec.LineSentence(args.input)

    if args.improve:
        # Load a previously trained model.
        model = gensim.models.word2vec.Word2Vec.load(args.output)
        # Remove the input suffix, if any.
        suffix = '.{0}'.format(model.iter)
        if args.output.endswith(suffix):
            args.output = args.output[:-len(suffix)]
        # Continue training.
        model.train(sentences)
    else:
        # Train a new model.
        model = gensim.models.word2vec.Word2Vec(
            sentences, size=args.dimension, window=args.max_distance,
            min_count=args.min_count, workers=args.workers,
            sg=1, hs=1, iter=args.num_epochs)

    # Save the model in a format suitable for further training.
    save_name = '{0}.{1}'.format(args.output, model.iter)
    model.save(save_name)


if __name__ == '__main__':
    main()
