#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import io
import os
import os.path

CORPUS_DIRECTORY='corpus'

# Maximum number of wikipedia articles to index per word. Can be
# overridden using the --max-size command-line argument.
max_index_size = 10000


def ingest(page, page_titles, depth=0, max_depth=1):

    title = page.title()
    if title in page_titles:
        return page_titles
    # Must be in one of the following namespaces:
    # Main, Category, Portal, Book.
    # https://en.wikipedia.org/wiki/Wikipedia:Namespace
    if page.namespace() not in (0, 14, 100, 108,):
        return page_titles
    page_titles.add(title)
    # Have we reached our target number of pages?
    if len(page_titles) >= max_index_size:
        raise StopIteration
    # Explore children of this page?
    if depth >= max_depth:
        return page_titles
    # Visit pages linked from this page.
    for sub_page in page.linkedPages(total=max_index_size // 3):
        ingest(sub_page, page_titles, depth + 1, max_depth)
    # Visit pages that refer to or embed this page.
    for sub_page in page.getReferences(total=max_index_size // 3):
        ingest(sub_page, page_titles, depth + 1, max_depth)

    return page_titles


def main():
    global max_index_size
    parser = argparse.ArgumentParser(
        description='Create an index for the training corpus.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default='words.txt',
                        help='Name of word list to use.')
    parser.add_argument('--index-size', type=int, default=max_index_size,
                        help='Target number of pages per word.')
    parser.add_argument('--encoding', type=str, default='utf8',
                        help='Encoding for saving corpus index.')
    args = parser.parse_args()

    max_index_size = args.index_size

    # Read the word list into memory and format using wikimedia conventions.
    # https://en.wikipedia.org/wiki/Wikipedia:Naming_conventions_(capitalization)
    with open(args.input, 'r') as f:
        words = [w.strip().capitalize() for w in f]
    print('Read {0} words from {1}.'.format(len(words), args.input))

    if not os.path.isdir(CORPUS_DIRECTORY):
        os.mkdir(CORPUS_DIRECTORY)

    # Use the english wikipedia with no user config and ignore warnings.
    os.environ['PYWIKIBOT2_NO_USER_CONFIG'] = '2'
    import pywikibot
    site = pywikibot.Site('en', 'wikipedia')

    for word in words:

        page_titles = set()
        try:
            # Try to ingest the page for this word directly.
            page = pywikibot.Page(site, word)
            page_titles = ingest(page, page_titles)

            # Try to ingest a disambiguation page for this word.
            if not page.isDisambig():
                page = pywikibot.Page(site, word + ' (disambiguation)')
                page_titles = ingest(page, page_titles)

            # Try to ingest the results of a site-wide search for this word.
            # Only include results in the Main namespace.
            results = site.search(
                searchstring=word, where='text', namespaces=[0])
            for page in results:
                page_titles = ingest(page, page_titles, depth=1)
        except StopIteration:
            # We normally get here once 10,000 pages have been ingested.
            pass

        # Save the set of all ingested page names for this word.
        out_name = os.path.join(CORPUS_DIRECTORY, '{0}.index'.format(word))
        with io.open(out_name, 'w', encoding=args.encoding) as out:
            for title in page_titles:
                out.write(title + '\n')

        print('Saved index of {0} pages to {1}.'
              .format(len(page_titles), out_name))


if __name__ == '__main__':
    main()
