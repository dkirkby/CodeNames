#!/usr/bin/env python
from __future__ import print_function, division

import argparse

import engine


def main():
    parser = argparse.ArgumentParser(
        description='Play the CodeNames game.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    e = engine.GameEngine()
    e.play_game()


if __name__ == '__main__':
    main()
