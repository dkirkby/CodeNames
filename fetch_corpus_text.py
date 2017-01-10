#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import io
import os.path
import warnings
import multiprocessing
import gzip
import random

import wikipedia

CORPUS_DIRECTORY='corpus'


def fetch(word, encoding='utf8', min_size=5e6):

    in_name = os.path.join(CORPUS_DIRECTORY, '{0}.index'.format(word))
    out_name = os.path.join(CORPUS_DIRECTORY, '{0}.txt.gz'.format(word))

    with io.open(in_name, 'r', encoding=encoding) as f_in:
        # Read all page titles.
        page_titles = [line.rstrip() for line in f_in]
        # Generate a random order of page titles.
        order = range(len(page_titles))
        random.shuffle(order)

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
                    except wikipedia.exceptions.DisambiguationError as e:
                        # Ignore disambiguation pages.
                        pass
                    except Exception as e:
                        print('Unexpected Error:: {0}'.format(e))

                # Save this article's content.
                f_out.write(content.encode(encoding))
                total_size += len(content)
                num_articles += 1
                if total_size >= min_size:
                    break

    return (word, len(page_titles), num_articles, total_size)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch indexed training corpus text.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default='words.txt',
                        help='Name of word list to use.')
    parser.add_argument('--encoding', type=str, default='utf8',
                        help='Encoding for corpus index and text files.')
    parser.add_argument('--nproc', type=int, default=20,
                        help='Number of processing pool workers to use.')
    args = parser.parse_args()

    # Read the word list into memory and format using wikimedia conventions.
    # https://en.wikipedia.org/wiki/Wikipedia:Naming_conventions_(capitalization)
    with open(args.input, 'r') as f:
        words = [w.strip().capitalize() for w in f]
    print('Read {0} words from {1}.'.format(len(words), args.input))

    pool = multiprocessing.Pool(processes=args.nproc)
    result = pool.map_async(fetch, words[:10])
    result.wait()
    for article_info in result.get():
        print(article_info)


if __name__ == '__main__':
    main()
