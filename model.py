from __future__ import print_function, division

import warnings

import nltk.stem.wordnet


class WordEmbedding(object):

    def __init__(self, filename='word2vec.dat.2'):
        # Import gensim here so we can mute a UserWarning about the Pattern
        # library not being installed.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            import gensim.models.word2vec

        # Load the model.
        self.model = gensim.models.word2vec.Word2Vec.load(filename)

        # Reduce the memory footprint since we will not be training.
        self.model.init_sims(replace=True)

        # Initialize a wordnet lemmatizer for stemming.
        self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


    def get_stem(self, word):
        """Return the stem of word.
        """
        if word in ('pass', 'passing', 'passed',):
            return 'pass'
        return self.lemmatizer.lemmatize(word).encode('ascii', 'ignore')


    def filter_clues(self, similar, vetos):
        """
        Filter a ranked list of (word, score) tuples to remove entries
        that are:
         - similar to words in the veto list
         - similar to previously selected clues
         - invalid clues.
        """
        veto_words = set(vetos)
        veto_stems = set([self.get_stem(word) for word in veto_words])
        filtered = []
        for word, score in similar:
            stem = self.get_stem(word)
            if stem in veto_stems:
                continue
            # Ignore words that are contained within a veto word or vice versa.
            contained = False
            for veto in veto_words:
                if veto in word or word in veto:
                    contained = True
                    break
            if contained:
                continue
            # Ignore words with special characters.
            if set(word) & set('\\.'):
                continue
            # Add this word to the veto list.
            veto_stems.add(stem)
            veto_words.add(word)
            filtered.append((word, score))

        return filtered


    def get_clues(self, words, vetos):
        """
        Return the best (word, score) clues for the specified words that
        are not similar to any of the specified vetos.
        """
        # Special handling for the word "march" which the embedding has trouble with.
        if len(words) > 1 and 'march' in words:
            return []

        return self.filter_clues(self.model.most_similar(words, topn=10), vetos)
