#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import gzip
import io
import multiprocessing
import os
import os.path
import random
import warnings

import wikipedia

from config import config

dry_run = False


def fetch(word, min_size=5e6):

    # Use a reproducible but different "random" shuffle for each word.
    random.seed(word)

    in_name = os.path.join(config.corpus_directory, config.template['index'].format(word))
    out_name = os.path.join(config.corpus_directory, config.template['articles'].format(word))

    # Has this word already been fetched?
    if os.path.exists(out_name):
        try:
            # Check the GZIP structure and size.
            with gzip.open(out_name, 'rb') as f_in:
                # Uncompress the whole file into memory.  This is relatively
                # expensive, but is the only foolproof check.
                content = f_in.read().decode(config.encoding)
                size = len(content)
                if size >= min_size:
                    return word, 0, 0, size
                print('Good file "{0}" below minimum size: {1} < {2}.'
                      .format(out_name, size, min_size))
        except Exception as e:
            print('Bad file "{0}":: {1}'.format(out_name, e))

    with io.open(in_name, 'r', encoding=config.encoding) as f_in:
        # Read all page titles.
        page_titles = [line.rstrip() for line in f_in]
        # Generate a random order of page titles.
        order = range(len(page_titles))
        random.shuffle(order)
        print('Fetching from {0} pages for {1}.'.format(len(page_titles), word))

        if dry_run:
            return word, 0, 0, 0

        total_size = 0
        num_articles = 0
        with gzip.open(out_name, 'wb') as f_out:

            for article_index in order:
                page_title = page_titles[article_index]
                # Read this article's plain-text content.
                with warnings.catch_warnings():
                    # Ignore warnings.  The expected warnings are:
                    # requests.packages.urllib3.exceptions.SubjectAltNameWarning
                    # UserWarning
                    warnings.simplefilter('ignore')
                    try:
                        page = wikipedia.page(
                            page_title, auto_suggest=False, preload=False)
                        content = page.content
                        # Save this article's content.
                        f_out.write(content.encode(config.encoding))
                        total_size += len(content)
                        num_articles += 1
                        if total_size >= min_size:
                            break
                    except wikipedia.exceptions.DisambiguationError:
                        # Ignore disambiguation pages.
                        pass
                    except Exception as e:
                        print('Unexpected Error:: {0}'.format(e))

    return word, len(page_titles), num_articles, total_size


def main():
    global dry_run
    parser = argparse.ArgumentParser(
        description='Fetch indexed training corpus text.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nproc', type=int, default=20,
                        help='Number of processing pool workers to use.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform a dry run only.')
    args = parser.parse_args()
    dry_run = args.dry_run

    # Read the word list into memory and format using wikimedia conventions.
    # https://en.wikipedia.org/wiki/Wikipedia:Naming_conventions_(capitalization)
    with open(config.word_list, 'r') as f:
        words = [w.strip().capitalize() for w in f]
    print('Read {0} words from {1}.'.format(len(words), config.word_list))

    pool = multiprocessing.Pool(processes=args.nproc)
    result = pool.map_async(fetch, words)
    result.wait()


if __name__ == '__main__':
    main()
