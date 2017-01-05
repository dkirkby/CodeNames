CodeNames AI
============

This is a project to study the performance of some standard natural-language and
machine-learning algorithms for replacing human players in the CodeNames game.

Requirements
------------

Code is currently developed and tested using python 2.7.  The following external
packages are required, and all available via pip:
 - wikipedia: interact with the wikimedia API.
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
corpus using:
```
./build_corpus.py
```
This will read the list of words used in the game from `words.txt` and writes plain text content downloaded from wikipedia into a single file per word in the `corpus/` subdirectory.

Next, use the following command to split the saved content into sentences and words, with punctuation removed and everything lower case:
```
./merge_corpus.py
```
The results are saved in a single compressed text file where the sentences for each topic
appear in random order.

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
nohup ./learn.py --workers 18 -n 100 > learn1.log &
nohup ./learn.py -o word2vec.dat.1 --workers 18 -n 100 --improve > learn2.log &
nohup ./learn.py -o word2vec.dat.2 --workers 18 -n 100 --improve > learn3.log &
nohup ./learn.py -o word2vec.dat.3 --workers 18 -n 100 --improve > learn4.log &
```

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
