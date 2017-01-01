#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import io
import os.path

import wikipedia


def main():
    parser = argparse.ArgumentParser(
        description='Build training corpus.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default='words.txt',
                        help='Name of word list to use.')
    parser.add_argument('--encoding', type=str, default='utf8',
                        help='Encoding for saving corpus text.')
    args = parser.parse_args()

    # Read the word list into memory and format using wikimedia conventions.
    # https://en.wikipedia.org/wiki/Wikipedia:Naming_conventions_(capitalization)
    with open(args.input, 'r') as f:
        words = [w.strip().capitalize() for w in f]
    print('Read {0} words from {1}.'.format(len(words), args.input))

    for word in words:
        titles = set()
        ##if word < 'Himalayas': continue
        print(word)

        # Look for a disambiguation page on this topic that lists pages
        # related to different meanings.
        try:
            page = wikipedia.page(
                word + ' (disambiguation)', preload=False,
                redirect=True, auto_suggest=False)
            # Assume that we have re-directed to a non-disambiguation page
            # if we get here (ie, with no DisambiguationError raised).
            titles.add(page.title)
        except wikipedia.exceptions.PageError as e:
            # Disambiguation page does not exist and does not redirect
            # so assume that there is a unique topic page.
            print('-> unique topic page')
            titles.add(unicode(word))
        except wikipedia.exceptions.DisambiguationError as e:
            # Record the list of pages related to this topic.
            titles |= set(e.options)

        # Do a search on this topic and add any new page titles it finds.
        try:
            titles |= set(wikipedia.search(word))
        except Exception as e:
            print('->' + str(e))

        num_pages = 0
        out_name = os.path.join('corpus', word + '.txt')
        with io.open(out_name, 'w', encoding=args.encoding) as out:
            for title in titles:
                # Remove any quotes.
                title = title.translate({ord('"'): None})
                print('  {0:3d} {1}'.format(
                    num_pages, title.encode('ascii', 'replace')))
                try:
                    page = wikipedia.page(
                        title, auto_suggest=False, redirect=True)
                    out.write(page.content)
                    num_pages += 1
                except wikipedia.exceptions.DisambiguationError as e:
                    # Skip second-level ambiguous topics.
                    print('.. ambiguous')
                except Exception as e:
                    print('!!' + str(e))
            print('  saved {0} pages to {1}'.format(num_pages, out_name))


if __name__ == '__main__':
    main()
