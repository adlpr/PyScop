# PyScop

PyScop is a (scop)[https://en.wikipedia.org/wiki/Scop] living in Python, who will craft Anglo-Saxon-style (alliterative verse)[https://en.wikipedia.org/wiki/Alliterative_verse] from your English-language input.

Originally written for use in (NaNoGenMo 2014)[https://github.com/dariusk/NaNoGenMo-2014/].

## Requirements

Requires (NLTK)[http://www.nltk.org/] and (espeak)[http://espeak.sourceforge.net/].

## Usage

To output an approximately 20-line (sometimes a tad more) poem using sentences contained in file `corpus.txt`:

```
import codecs, nltk
from pyscop import PyScop

corpus = codecs.open('corpus.txt', 'r', 'utf-8').read()
sentences = nltk.sent_tokenize(corpus)
scop = PyScop(sentences)
print scop.generate_poem(20)
```