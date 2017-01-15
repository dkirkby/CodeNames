from __future__ import print_function, division

import warnings

import numpy as np

import nltk.stem.wordnet

import sklearn.cluster


class WordEmbedding(object):

    def __init__(self, filename):
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
        # Hardcode some stemming rules for the default CodeName words
        # that the wordnet lemmatizer doesn't know about.
        if word in ('pass', 'passing', 'passed',):
            return 'pass'
        if word in ('microscope', 'microscopy'):
            return 'microscope'
        if word in ('mexico', 'mexican', 'mexicans', 'mexicali'):
            return 'mexico'
        if word in ('theater', 'theatre', 'theaters', 'theatres',
                    'theatrical', 'theatricals'):
            return 'theater'
        if word in ('alp', 'alps', 'apline', 'alpinist'):
            return 'alp'
        return self.lemmatizer.lemmatize(word).encode('ascii', 'ignore')


    def get_clue(self, clue_words, pos_words, neg_words, veto_words,
                 veto_margin=0.2, num_search=100, verbose=0):
        """
        """
        if verbose >= 2:
            print('CLUE:', clue_words)
            print(' POS:', pos_words)
            print(' NEG:', neg_words)
            print('VETO:', veto_words)

        # Initialize the list of illegal clues.
        illegal_words = list(pos_words) + list(neg_words) + list(veto_words)
        illegal_stems = set([self.get_stem(word) for word in illegal_words])

        # Get the internal indices and normalized vectors for each word.
        clue_indices = [self.model.vocab[word].index for word in clue_words]
        clue_vectors = self.model.syn0norm[clue_indices]
        pos_indices = [self.model.vocab[word].index for word in pos_words]
        pos_vectors = self.model.syn0norm[pos_indices]
        neg_indices = [self.model.vocab[word].index for word in neg_words]
        neg_vectors = self.model.syn0norm[neg_indices]
        veto_indices = [self.model.vocab[word].index for word in veto_words]
        veto_vectors = self.model.syn0norm[veto_indices]

        # Find the normalized mean of the words in the clue group.
        mean_vector = clue_vectors.mean(axis=0)
        mean_vector /= np.sqrt(mean_vector.dot(mean_vector))

        # Calculate the cosine distances between the mean vector and all
        # the words in our vocabulary.
        cosines = np.dot(self.model.syn0norm[:, np.newaxis],
                         mean_vector).reshape(-1)

        # Sort the vocabulary by decreasing cosine similarity with the mean.
        closest = np.argsort(cosines)[::-1]

        # Select the clue whose minimum cosine from the words is largest
        # (i.e., smallest maximum distance).
        best_clue = None
        max_min_cosine = -2.
        for i in range(num_search):
            clue_index = closest[i]
            clue = self.model.index2word[clue_index]
            # Ignore clues with the same stem as an illegal clue.
            if self.get_stem(clue) in illegal_stems:
                continue
            # Ignore clues that are contained within an illegal clue or
            # vice versa.
            contained = False
            for illegal in illegal_words:
                if clue in illegal or illegal in clue:
                    contained = True
                    break
            if contained:
                continue
            # Calculate the cosine similarity of this clue with all of the
            # positive, negative and veto words.
            clue_vector = self.model.syn0norm[clue_index]
            clue_cosine = np.dot(clue_vectors[:, np.newaxis], clue_vector)
            neg_cosine = np.dot(neg_vectors[:, np.newaxis], clue_vector)
            veto_cosine = np.dot(veto_vectors[:, np.newaxis], clue_vector)
            # Is this closer to all of the positive words than our previous best?
            min_clue_cosine = np.min(clue_cosine)
            if min_clue_cosine < max_min_cosine:
                continue
            # Are all positive words more similar than any negative words?
            if list(neg_words):
                max_neg_cosine = np.max(neg_cosine)
                if max_neg_cosine >= min_clue_cosine:
                    # A negative word is likely to be selected before all the
                    # positive words.
                    if verbose >= 3:
                        neg_word = neg_words[np.argmax(neg_cosine)]
                        print('neg word {0} is a distractor (cosine={1:.4f})'
                              .format(neg_word, max_neg_cosine))
                    continue
            # Is this word too similar to any of the veto words?
            if list(veto_words):
                max_veto_cosine = np.max(veto_cosine)
                if max_veto_cosine >= min_clue_cosine - veto_margin:
                    # A veto word is too likely to be selected before all the
                    # positive words.
                    if verbose >= 2:
                        veto_word = veto_words[np.argmax(veto_cosine)]
                        print('veto word {0} is a distractor (cosine={1:.4f})'
                              .format(veto_word, max_veto_cosine))
                    continue
            # If we get here, we have a new best clue.
            max_min_cosine = min_clue_cosine
            best_clue = clue
            if verbose >= 1:
                words = [w.upper() for w in clue_words]
                print('{0} = {1} (min_cosine={2:.4f})'
                      .format('+'.join(words), clue, min_clue_cosine))

        return best_clue, max_min_cosine


    def get_clusters_kmeans(self, words):
        """Use the KMeans algorithm to find word clusters.
        """
        words = np.asarray(words)
        num_words = len(words)
        X = np.empty((num_words, self.model.vector_size))
        for i, word in enumerate(words):
            X[i] = self.model.syn0norm[self.model.vocab[word].index]

        for num_clusters in range(1, num_words):
            kmeans = sklearn.cluster.KMeans(num_clusters).fit(X)
            for label in set(kmeans.labels_):
                members = words[kmeans.labels_ == label]
                print('{0},{1}: {2}'.format(num_clusters, label, members))


    def get_clusters_dbscan(self, words, min_sep=1.25):
        """Use the DBSCAN algorithm to find word clusters.
        """
        # Calculate the distance matrix for the specified words.
        words = np.asarray(words)
        num_words = len(words)
        distance = np.zeros((num_words, num_words))
        for i1 in range(num_words):
            for i2 in range(i1):
                cosine = self.model.similarity(words[i1], words[i2])
                distance[i1, i2] = distance[i2, i1] = np.arccos(cosine)

        # Initailize cluster finder.
        db = sklearn.cluster.DBSCAN(
            eps=min_sep, min_samples=1, metric='precomputed', n_jobs=1)
        db.fit(distance)
        for label in set(db.labels_):
            members = words[db.labels_ == label]
            print('{0}: {1}'.format(label, members))
