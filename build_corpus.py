#!/usr/bin/env python
from __future__ import print_function, division

import argparse

import wikipedia

# Himalayas -> no disambiguation page found
# Ice Cream -> no disambiguation page found
# Limousine -> no disambiguation page found
# Microscope -> no disambiguation page found
# Pants -> no disambiguation page found
# Ruler -> no disambiguation page found
# Shoe -> show
# Stream -> street

def main():
    parser = argparse.ArgumentParser(
        description='Build training corpus.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default='words.txt',
                        help='Name of word list to use.')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        words = [w.strip() for w in f]
    print('Read {0} words from {1}.'.format(len(words), args.input))

    for word in words:
        word = ' '.join([w.capitalize() for w in word.split()])
        print(word)
        # Lookup the disambiguation pages for this topic.
        page_name = word + ' (disambiguation)'
        try:
            page = wikipedia.page(
                page_name, preload=False, redirect=True, auto_suggest=False)
            # Assume that we have re-directed to a non-disambiguation page.
            pages = [page.title]
        except wikipedia.exceptions.PageError as e:
            # Redirect page does not exist so use topic directly.
            pages = [word]
        except wikipedia.exceptions.DisambiguationError as e:
            pages = e.options
        num_pages = 0
        for page in pages:
            # Skip second-level disambiguation pages.
            if page.endswith(' (disambiguation)'):
                continue
            #print('  ' + page)
            num_pages += 1
        print('  {0}'.format(num_pages))


if __name__ == '__main__':
    main()
