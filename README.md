CodeNames AI
============

This is a project to study the performance of some standard natural-language and
machine-learning algorithms for replacing human players in the CodeNames game.

Requirements
------------

Code is currently developed and tested using python 2.7.  The following external
packages are required, and all available via pip:
 - pywikibot: interact with the wikimedia API.
 - nltk: Natural language tool kit.
 - gensim: Learn word vector embeddings from a large corpus text.

After `nltk` is installed, you will need to download the following data files:
- Models / punkt (13.0Mb)
- Corpora / wordnet (10.3Mb)

Use the following python commands to open the NLTK download dialog:
```
import nltk
nltk.download()
```

Knowledge Base
--------------

The AI learns from a large set of sentences called the "corpus". Build the
corpus by first selecting articles from the english wikipedia related to each
word with:
```
./create_corpus_index.py
```
This will read the list of words used in the game from `words.txt` and writes an
index file `corpus/<Word>.index` for each one containing wikipedia page titles (with
utf-8 encoding).  This script aims for the same number (nominally 10K) of articles
for each word, but some words might have less if not enough articles can be found.
I had to restart this process a few times due to uncaught pywikibot.data.api.APIError
exceptions (TODO: implement automatic recovery from these).  The whole index-creation
step takes about 6 hours.

The next step is to download the content of all articles using:
```
./fetch_corpus_text.py
```
This step is performed in parallel (using 20 processes by default) since it is IO bound.
In case it fails for some reason, it can be restarted and will automatically skip over
any words that have already been fetched.  The goal of the fetching step is to download
~5M characters of plain (unicode) text for each code word.  This does not require
downloading all of the indexed articles, so articles are processed in a random (but
reproducible) order until at least 5M characters have been downloaded.

The final step is to preprocess the downloaded text into a format suitable for
feeding directly into word2vec training:
- Split into sentences, one per line.
- Remove markup headings (e.g., "== References ==").
- Split into word tokens.
- Combine compound words from the word list ("ice cream" -> "ice_cream").
- Remove punctuation.
- Convert to lower case.

This step is performed using:
```
./preprocess_corpus.py
```
which converts each `corpus/Word.txt.gz` into a corresponding `corpus/Word.pre.gz`.
Processing runs in a single process since it is relatively fast (~90 mins) and this
simplifies collecting the summary statistics in the `freqs.dat` output file.

Machine Learning
----------------

Machine learning consists of iteratively finding an embedding of all words into a
300-dimensional vector space that captures important aspects of their semantic
relationships. Run training with the command:
```
./learn.py
```
In practice, this takes a while (~30 hours total) so we run five consecutive
passes using 20 cores in parallel:
```
nohup ./learn.py --workers 20 --npass 1 > learn1.log &
nohup ./learn.py --workers 20 --npass 2 > learn2.log &
nohup ./learn.py --workers 20 --npass 3 > learn3.log &
nohup ./learn.py --workers 20 --npass 4 > learn4.log &
nohup ./learn.py --workers 20 --npass 5 > learn5.log &
```
Each pass starts with a random shuffle of the corpus followed by 5 epochs of training,
with a learning rate that decreases linearly from 0.0251 to 0.0001 over the 5 passes.
The output from each job consists of 4 files (with N = 1-5):
- word2vec.dat.N
- word2vec.dat.N.syn0.npy
- word2vec.dat.N.syn1.npy
- word2vec.dat.N.syn1neg.npy

The first file contains the vocabulary words and corresponding embedded vectors.
The last three files contain the neural network weights.

After training, run an evaluation suite of the embedding quality using:
```
./evaluate.py -i word2vec.dat.4 --top-singles 10 --top-pairs 10 --save-plots
```

Play
----

Finally, try playing the game with an AI spymaster for both teams using:
```
./play.py --config CHCH --seed 123
```
