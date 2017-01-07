from __future__ import print_function, division

import sys
import re
import itertools

import numpy as np

import model


class GameEngine(object):

    def __init__(self, seed=None, init=None, wordlist='words.txt',
                 model_name='word2vec.dat'):

        # Load our word list if necessary.
        # TODO: Max length of 11 is hardcoded here and in print_board()
        self.words = np.empty(400, dtype='S11')
        with open(wordlist) as f:
            for i, line in enumerate(f.readlines()):
                self.words[i] = line.rstrip().lower().replace(' ', '_')

        # Initialize our word embedding model if necessary.
        self.model = model.WordEmbedding(model_name)

        # Initialize random numbers.
        self.generator = np.random.RandomState(seed=seed)

        # Useful regular expressions.
        self.valid_clue = re.compile('^([a-zA-Z]+) ([0-9])$')


    def initialize_random_game(self, size=5):

        self.size = size

        # Shuffle the wordlist.
        shuffle = self.generator.choice(
            len(self.words), size * size, replace=False)
        self.board = self.words[shuffle]

        # Specify the layout for this game.
        assignments = self.generator.permutation(size * size)
        self.owner = np.empty(size * size, int)
        self.owner[assignments[0]] = 0 # assassin
        self.owner[assignments[1:10]] = 1 # first player
        self.owner[assignments[10:19]] = 2 # second player
        self.owner[assignments[19:]] = 3 # bystander

        # All cards are initially visible.
        self.visible = np.ones_like(self.owner, dtype=bool)
        self.num_turns = 0


    def initialize_from_words(self, initial_words, size=5):
        """
        The initial_words parameter should be in the format:

            ASSASSIN;TEAM1;TEAM2;NEUTRAL

        where each group consists of comma-separated words from the word list.

        The total number of words must be <= size * size. Any missing words
        are considered to be already covered and neutral.
        """
        self.size = size

        word_groups = initial_words.split(';')
        if len(word_groups) != 4:
            raise ValueError('Expected 4 groups separated by semicolon.')

        board, owner, visible = [], [], []
        for group_index, word_group in enumerate(word_groups):
            words = word_group.split(',')
            for word in words:
                word = word.lower().replace(' ', '_')
                if word not in self.words:
                    raise ValueError('Invalid word "{0}".'.format(word))
                if word in board:
                    raise ValueError('Duplicate word "{0}".'.format(word))
                board.append(word)
                owner.append(group_index)
                visible.append(True)
        if len(board) > size * size:
            raise ValueError(
                'Too many words. Expected <= {0}.'.format(size * size))
        # Add dummy hidden words if necessary.
        while len(board) < size * size:
            board.append('---')
            owner.append(3)
            visible.append(False)

        self.board = np.array(board)
        self.owner = np.array(owner)
        self.visible = np.array(visible)

        # Perform a random shuffle of the board.
        shuffle = self.generator.permutation(size * size)
        self.board = self.board[shuffle]
        self.owner = self.owner[shuffle]
        self.visible = self.visible[shuffle]

        # TEAM1 plays next.
        self.num_turns = 0


    def print_board(self, spymaster=False):

        board = self.board.reshape(self.size, self.size)
        owner = self.owner.reshape(self.size, self.size)
        visible = self.visible.reshape(self.size, self.size)

        for row in range(self.size):
            for col in range(self.size):
                word = board[row, col]
                tag = '#<>-'[owner[row, col]]
                if not visible[row, col]:
                    word = tag * 11
                elif not spymaster:
                    tag = ' '
                if not spymaster or owner[row, col] in (0, 1, 2):
                    word = word.upper()
                sys.stdout.write('{0}{1:11s} '.format(tag, word))
            sys.stdout.write('\n')


    def play_computer_spymaster(self, gamma=1.0, verbose=True):

        self.print_board(spymaster=True)
        print('Thinking...')
        sys.stdout.flush()

        player = self.num_turns % 2
        player_label = '<>'[player] * 3
        player_words = self.board[(self.owner == player + 1) & self.visible]
        avoid_words = self.board[
            (self.owner > 0) & (self.owner != player + 1) & self.visible]
        veto_words = self.board[(self.owner == 0) & self.visible]

        # Loop over all permutations of words.
        num_words = len(player_words)
        best_score, saved_clues = [], []
        for count in range(num_words, 0, -1):
            # Multiply similarity scores by this factor for any clue
            # corresponding to this many words.
            bonus_factor = count ** gamma
            for group in itertools.combinations(range(num_words), count):
                words = player_words[list(group)]
                clue, score = self.model.get_clue(
                    words, player_words, avoid_words, veto_words)
                if clue:
                    best_score.append(score * bonus_factor)
                    saved_clues.append((clue, words))
        num_clues = len(saved_clues)
        order = sorted(xrange(num_clues),
                       key=lambda k: best_score[k], reverse=True)
        if verbose:
            for i in order[:10]:
                clue, words = saved_clues[i]
                print('{0:.3f} {1} = {2}'.format(
                    best_score[i], ' + '.join([w.upper() for w in words]), clue))

        clue, words = saved_clues[order[0]]
        return clue, len(words)


    def play_human_spymaster(self):

        self.print_board(spymaster=True)

        player = self.num_turns % 2
        player_label = '<>'[player] * 3
        while True:
            try:
                clue = raw_input('{0} Enter your clue: '.format(player_label))
            except KeyboardInterrupt:
                print('\nBye.')
                sys.exit(0)
            matched = self.valid_clue.match(clue)
            if matched:
                word, count = matched.groups()
                count = int(count)
                return word, count
            print('Invalid clue, should be WORD COUNT.')


    def play_human_team(self, word, count):

        player = self.num_turns % 2
        player_label = '<>'[player] * 3
        player_words = set(
            self.board[(self.owner == player + 1) & self.visible])
        assasin = self.board[self.owner == 0]

        num_guesses = 0
        while num_guesses <= count + 1:
            self.print_board(spymaster=False)
            print('{0} your clue is: {1} {2}'.format(player_label, word, count))
            num_guesses += 1

            while True:
                try:
                    guess = raw_input('{0} enter your guess #{1}: '
                                      .format(player_label, num_guesses))
                except KeyboardInterrupt:
                    print('\nBye.')
                    sys.exit(0)
                guess = guess.strip().lower().replace(' ', '_')
                if guess == '' or guess in self.board[self.visible]:
                    break
                print('Invalid guess, should be a visible word.')

            if guess == '':
                # Team does not want to make any more guesses.
                return True

            loc = np.where(self.board == guess)[0]
            self.visible[loc] = False

            if guess == assasin:
                print('{0} You guessed the assasin - game over!'
                      .format(player_label))
                return False

            if guess in player_words:
                player_words.remove(guess)
                if player_words:
                    print('{0} Congratulations, keep going!'
                          .format(player_label))
                else:
                    print('{0} You won!!!'.format(player_label))
                    return False
            else:
                print('{0} Sorry!'.format(player_label))
                break

        return True


    def play_turn(self, spymaster='human', team='human'):

        if spymaster == 'human':
            word, count = self.play_human_spymaster()
        else:
            word, count = self.play_computer_spymaster()

        if team == 'human':
            ongoing = self.play_human_team(word, count)
        else:
            raise NotImplementedError()

        self.num_turns += 1
        return ongoing


    def play_game(self, spymaster1='human', team1='human',
                  spymaster2='human', team2='human', init=None):

        if init is None:
            self.initialize_random_game()
        else:
            self.initialize_from_words(init)
        while True:
            if not self.play_turn(spymaster1, team1): break
            if not self.play_turn(spymaster2, team2): break
