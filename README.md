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
In practice, this takes a while so I run several iterations using multiple cores in parallel, and split the work up into ~8 hour chunks:
```
nohup ./learn.py --workers 18 --npass 1 > learn1.log &
nohup ./learn.py --workers 18 --npass 2 > learn2.log &
nohup ./learn.py --workers 18 --npass 3 > learn3.log &
```
Each job performs a random shuffle of the corpus followed by 10 epochs of training.
The output from each job consists of 4 files (with N = npass):
- word2vec.dat.N
- word2vec.dat.N.syn0.npy
- word2vec.dat.N.syn1.npy
- word2vec.dat.N.syn1neg.npy

The first file contains the vocabulary words and corresponding embedded vectors.
The last three files contain the neural network weights.

After training, run an evaluation suite of the embedding quality using, e.g.
```
./evaluate.py -i word2vec.dat.4 --top-singles 20 --top-pairs 20 --save-plots
```
Results for single-word matches after the first job (10 epochs) are:
```
0.972 MARCH = april
0.931 FLUTE = clarinet
0.910 PIANO = violin
0.891 GOLD = silver
0.867 PISTOL = semi-automatic
0.852 CHOCOLATE = caramel
0.848 PANTS = trousers
0.845 MISSILE = surface-to-air
0.843 BERLIN = munich
0.840 TOKYO = osaka
0.837 DEGREE = bachelor
0.830 WHALE = humpback
0.828 CHURCH = episcopal
0.828 KETCHUP = mayonnaise
0.824 COURT = supreme
0.823 SERVER = client
0.823 THUMB = finger
0.821 JUPITER = neptune
0.814 GERMANY = austria
0.814 DISEASE = infection
```
then after four jobs (40 epochs):
```
0.975 MARCH = april
0.909 FLUTE = clarinet
0.906 PIANO = violin
0.888 GOLD = silver
0.883 CHOCOLATE = caramel
0.853 WHALE = humpback
0.849 TOKYO = osaka
0.848 PANTS = shirt
0.831 DEGREE = bachelor
0.829 MISSILE = ballistic
0.828 CHINA = taiwan
0.826 PISTOL = rifle
0.826 EMBASSY = consulate
0.823 BERLIN = munich
0.822 GERMANY = austria
0.821 KETCHUP = mayonnaise
0.817 CZECH = slovak
0.814 COPPER = zinc
0.809 DRESS = attire
0.809 JUPITER = uranus
```
Most of the code words are the same, but with some different clues, e.g.
- PISTOL: semi-automatic -> rifle
- PANTS: trousers -> shirt
- MISSILE: surface-to-air -> ballistic
- JUPITER: neptune -> uranus

For pairs, the pass-1 results are:
```
0.849 PIANO + FLUTE = cello
0.805 PANTS + DRESS = trousers
0.755 LEMON + CHOCOLATE = vanilla
0.751 GERMANY + FRANCE = belgium
0.750 HORSESHOE + BAT = rhinolophus
0.729 STRING + PIANO = quartet
0.723 ICE_CREAM + CHOCOLATE = candy
0.719 PASTE + KETCHUP = garlic
0.718 WEB + SERVER = browser
0.709 TURKEY + GREECE = cyprus
0.707 HOTEL + CASINO = resort
0.703 ORGAN + FLUTE = harpsichord
0.703 PIANO + ORGAN = harpsichord
0.699 RABBIT + DOG = cat
0.696 PIANO + HORN = flute
0.690 STRING + FLUTE = violin
0.686 SCHOOL + DEGREE = graduate
0.679 HORN + FLUTE = trumpet
0.678 MOON + JUPITER = venus
0.672 GERMANY + CZECH = poland
```
and after pass-4:
```
0.826 PIANO + FLUTE = cello
0.799 PANTS + DRESS = trousers
0.748 HORSESHOE + BAT = rhinolophus
0.742 ICE_CREAM + CHOCOLATE = candy
0.733 GERMANY + FRANCE = belgium
0.733 LEMON + CHOCOLATE = vanilla
0.711 PASTE + KETCHUP = sauce
0.707 TURKEY + GREECE = cyprus
0.705 GERMANY + CZECH = hungary
0.701 EUROPE + AFRICA = asia
0.701 PIANO + ORGAN = harmonium
0.694 STRING + PIANO = quartet
0.694 GREECE + FRANCE = italy
0.694 GREECE + GERMANY = italy
0.692 ORGAN + FLUTE = harmonium
0.688 STRING + FLUTE = violin
0.686 RABBIT + DOG = cat
0.678 WEB + SERVER = browser
0.676 LEMON + ICE_CREAM = flavored
0.676 CHEST + ARM = shoulder
```
Again, many of the code words are the same with some new clues:
- PASTE + KETCHUP: garlic -> sauce
- ORGAN + FLUTE: harpsichord -> harmonium
- PIANO + ORGAN: harpsichord -> harmonium
- GERMANY + CZECH: poland -> hungary

Play
----

Finally, try playing the game with an AI spymaster for both teams using:
```
./play.py --config CHCH --seed 123
```
