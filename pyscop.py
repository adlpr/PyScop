# PyScop: An Anglo-Saxon scop living in Python.

import re, random
from nltk.corpus import cmudict

from subprocess import check_output

try: import simplejson as json
except ImportError: import json

CMUDICT = cmudict.dict()
try: GENPHON = json.loads(open('genphon.json', 'r').read())
except: GENPHON = json.loads('{}')

# common words always treated as unstressed.
STOPWORDS = open('stopwords.txt', 'r').read().split('\n')

class PyScop():
  def __init__(self, corpus):
    """Takes corpus: a list of sentence strings (untokenized)."""
    # tokenize by word (custom)
    self.sents = [custom_word_tokenize(sentence) for sentence in corpus]
    self.verses = []
    # add valid verses for use in poem generation
    for s in [ScopSentence(s) for s in self.sents]:
      v = get_verse(s)
      if v and (v.can_be_a or v.can_be_b):
        self.verses.append(v)

  def generate_poem(self, target_lines=20):
    """Return a formatted string containing a generated poem."""
    assert target_lines > 2
    # copy this (shallow, not going to be changing the objects, just popping)
    verses = list(self.verses)
    random.shuffle(verses)

    # start generating poem
    current_halfline = 'a'
    assembled_verses = []
    out_of_options = False

    while verses and not (sum([len(v.halflines) for v in assembled_verses])/2 >= target_lines
                            and current_halfline == 'a'):
      found_one = False
      # randomize a forced internal-alliteration
      if [v for v in verses if v.can_be_a and v.alliterates_internally[0]]:
        force_ia = random.random() > 0.8 # force it 20% of the time, if possible
      else: force_ia = False

      for i in xrange(len(verses)):
        # start by finding a verse that can start at current halfline.
        # case 'a'
        if current_halfline == 'a':
          if verses[i].can_be_a and not (force_ia and not verses[i].alliterates_internally[0]):
            found_one = True
            matching_verse = verses.pop(i)
            assembled_verses.append(matching_verse)
            # calc new current_halfline: if odd # hls then b, if even a
            current_halfline = 'ab'[len(matching_verse.halflines) % 2 == 1]
            break
        # case 'b'
        elif current_halfline == 'b' and verses[i].can_be_b:
          # check alliteration
          sia1, sia2 = assembled_verses[-1].halflines[-1].si1, assembled_verses[-1].halflines[-1].si2
          sib1 = verses[i].halflines[0].si1
          if alliterates(sia1, sib1) or alliterates(sia2, sib1):
            found_one = True
            matching_verse = verses.pop(i)
            assembled_verses.append(matching_verse)
            # calc new current_halfline: if even # hls then b, if odd a
            current_halfline = 'ab'[len(matching_verse.halflines) % 2 == 0]
            break

      if not found_one:
        # try discarding the last match.
        popped_verse = assembled_verses.pop()
        # reset halfline
        current_halfline = 'ab'[len(popped_verse.halflines) % 2 == [0, 1][current_halfline == 'a']]
        # if there aren't any potential matches left, just give up
        if not [v for v in verses if [v.can_be_b, v.can_be_a][current_halfline == 'a']]:
          return None

    return self.format_poem(assembled_verses)

  def format_poem(self, verses):
    halflines = []
    for hl in [v.halflines for v in verses]:
      halflines += hl
    surface_texts = [hl.surface_text for hl in halflines]
    assert len(surface_texts) % 2 == 0
    surface_texts_paired = [(surface_texts[i], surface_texts[i+1]) for i in xrange(0, len(surface_texts), 2)]
    return '\n'.join(['\t'.join(hl) for hl in surface_texts_paired])

  def percentage_valid(self):
    """Returns percentage of valid verses in corpus, for measurement/comparison purposes."""
    return float(len(self.verses)) / len(self.sents)


class ScopVerse:
  def __init__(self, s, splits):
    """Takes s: a ScopSentence; splits: a list of indices to split the sentence into halflines"""
    self.s = s # source sentence
    # split it into halflines
    self.halflines = [ScopHalfline(s, start, end) for start, end in zip(splits[:-1], splits[1:])]
    self.alliterates_internally = [hl.alliterates_internally() for hl in self.halflines]
    self.can_be_a = self.can_be_a()
    self.can_be_b = self.can_be_b()

  # if any halfline in verse alliterates internally, that determines
  #  the pattern, so test that first.

  def can_be_a(self):
    if len(self.halflines) == 1: return True
    if True in self.alliterates_internally:
      for i,ai in enumerate(self.alliterates_internally):
        # must all be even indices
        if ai and i%2 == 1:
          return False
    for i in xrange(0, len(self.halflines), 2):
      # [0] si1 or si2 == [1] si1, != si2
      # [2] si1 or si2 == [3] si1, != si2 etc.
      hl = self.halflines
      if i+1 != len(self.halflines): # if this isn't the last line
        if not ((alliterates(hl[i+1].si1, hl[i].si1) or alliterates(hl[i+1].si1, hl[i].si2)) and
                (not alliterates(hl[i+1].si2, hl[i].si1) and not alliterates(hl[i+1].si2, hl[i].si2))):
          return False
    return True # tests passed

  def can_be_b(self):
    if len(self.halflines) == 1:
      if self.alliterates_internally[0]: return False
      else: return True
    if True in self.alliterates_internally:
      for i,ai in enumerate(self.alliterates_internally):
        # must all be odd indices
        if ai and i%2 == 0:
          return False
    for i in xrange(1, len(self.halflines), 2):
      # [1] si1 or si2 == [2] si1, != si2
      # [3] si1 or si2 == [4] si1, != si2 etc.
      hl = self.halflines
      if i+1 != len(self.halflines): # don't check after the last line
        if not ((alliterates(hl[i+1].si1, hl[i].si1) or alliterates(hl[i+1].si1, hl[i].si2)) and
                (not alliterates(hl[i+1].si2, hl[i].si1) and not alliterates(hl[i+1].si2, hl[i].si2))):
          return False
    return True # tests passed


class ScopHalfline:
  def __init__(self, s, start, end):
    """Takes s: a ScopSentence object; start, end: indexes of halfline within sentence"""
    self.s = s # source sentence
    self.tokens = s.tokens[start:end]
    self.surface_text = untokenize(self.tokens)
    self.words = s.words[start:end]
    self.phonetic = s.phonetic[start:end]
    self.stress = s.stress[start:end]
    self.si1, self.si2 = self.get_stressed_initials()

  def get_stressed_initials(self):
    """Returns an ordered pair of arpabet phones."""
    stressed_words_phonetic = [p for i,p in enumerate(self.phonetic) if '1' in self.stress[i]]
    assert len(stressed_words_phonetic) == 2
    # rules are: use only the first, unless s is followed by t, p or k
    cons = []
    for word in stressed_words_phonetic:
      initial = word[0]
      if initial == 'S':
        secondary = word[1]
        if secondary == 'T' or secondary == 'P' or secondary == 'K':
          cons.append(initial+secondary)
        else:
          cons.append(initial)
      else:
        cons.append(initial)
    return cons[0], cons[1]

  def alliterates_internally(self):
    return alliterates(self.si1, self.si2)


class ScopSentence:
  def __init__(self, tokens):
    """Takes tokens: a list representing a tokenized sentence."""
    self.tokens = tokens # raw tokens
    self.words = [ScopWord(token) for token in tokens] # word objects
    self.phonetic = self.to_phonetic()
    self.stress = self.to_stress_pattern()

  def to_phonetic(self):
    return [word.phonetic for word in self.words]

  def to_stress_pattern(self):
    return [word.stress for word in self.words]


UNPRONOUNCEABLE = re.compile(r'([^\w\s\d&]+|[<xX:]3|D[:xX])')

class ScopWord:
  def __init__(self, string):
    self.string = string
    self.lowercase = string.lower()
    self.phonetic = self.to_phonetic()
    self.stress = self.to_stress_pattern()

  def to_phonetic(self):
    # check if actually a pronounceable word, first.
    m = UNPRONOUNCEABLE.search(self.string)
    if m and m.group() == self.string:
      return []
    if self.lowercase in CMUDICT:
      return CMUDICT[self.lowercase][0] # top match
    elif self.lowercase in GENPHON:
      return GENPHON.get(self.lowercase) # top match
    else:
      # try removing unpronouncable parts
      newstring = re.sub(UNPRONOUNCEABLE.pattern, '', self.lowercase).strip()
      if newstring != self.lowercase:
        if newstring in CMUDICT:
          return CMUDICT[newstring][0]
        elif newstring in GENPHON:
          return GENPHON.get(newstring)
      # ok, not found. use espeak as backup.
      phones = check_output(['espeak', '-q','--ipa=3', '-v', 'en-us', newstring.encode('utf-8')])
      phones = phones.decode('utf-8').strip().replace(u'ː', '').replace(' ', '_').split('_')
      #print 'new word:', newstring.encode('utf-8')
      GENPHON[self.lowercase] = [ipa_to_arpabet(phone) for phone in phones]
      json.dump(GENPHON, open('genphon.json', 'w'))
      return GENPHON[self.lowercase]

  def to_stress_pattern(self):
    # treats 2 as 0. simplifies because I don't have to figure out alliteration
    # within a single word (no syllable boundaries required). maybe someday?
    if self.lowercase.replace("'",'') in STOPWORDS:
      return ['0' for vowel in [phoneme for phoneme in self.phonetic if is_vowel(phoneme)]]
    else:
      pattern = [['0', '1'][vowel[-1] == '1'] for vowel in [phoneme for phoneme in self.phonetic if is_vowel(phoneme)]]
      # nuke any additional stresses after the first just to be sure
      # (happens in some compound words like 'baseball')
      found_first = False
      for i in xrange(len(pattern)):
        if pattern[i] == '1':
          if found_first:
            pattern[i] = '0'
          else:
            found_first = True
      return pattern


def get_verse(s):
  """Takes s: ScopSentence object; returns ScopVerse object if the sentence has a valid stress pattern"""
  # stress pattern for sentence is converted into single string
  # word boundaries kept with commas
  stress_str = ','.join([''.join(word) for word in s.stress]) + ','
  # valid halflines:  / x / x   x / x /   x / / x
  re_halfline = re.compile(r'(,*1,*0?,*0?,*0?,*0?,*1,*0?,*0?,*0?,*0?,+|\
                              ,*0?,*0?,*0?,*0?,*1,*0?,*0?,*0?,*0?,*1,+|\
                              ,*0?,*0?,*0?,*0?,*1,*1,*0?,*0?,*0?,*0?,+)')
  m = re_halfline.findall(stress_str)
  # if matches found span the entire sentence, we have a winner
  if ''.join(m) == stress_str:
    # step through the sentence to get split indices
    splits = [0]
    current_index = 0
    for match in m:
      current_index += len(match.split(','))-1
      splits.append(current_index)
    return ScopVerse(s, splits)
  else:
    return None


def custom_word_tokenize(text):
  """Adapted from nltk's treebank tokenizer with e.g. contraction splitting removed,"""

  # starting quotes
  #text = re.sub(r'^\"', r'“', text)
  text = re.sub(r'^(\"|“)', r'\1', text)
  #text = re.sub(r'(``)', r' \1 ', text)
  text = re.sub(r'(``|“)', r' \1 ', text)
  #text = re.sub(r'([ (\[{<])"', r'\1 `` ', text)
  text = re.sub(r'([ (\[{<])("|“)', r'\1 \2 ', text)

  # punctuation
  text = re.sub(r'([:,])([^\d])', r' \1 \2', text)
  text = re.sub(r'\.\.\.', r' ... ', text)
  text = re.sub(r'[;@#$%&/-]', r' \g<0> ', text)
  text = re.sub(r'([^\.])(\.)([\]\)}>"\']*)\s*$', r'\1 \2\3 ', text)
  text = re.sub(r'[?!]', r' \g<0> ', text)

  text = re.sub(r"([^'])' ", r"\1 ' ", text)

  # parens, brackets, etc.
  text = re.sub(r'[\]\[\(\)\{\}\<\>]', r' \g<0> ', text)
  text = re.sub(r'--', r' -- ', text)

  # ADDED: treat hyphenated words as separate
  text = re.sub(r'([\w\d])-([\w\d])', r'\1 - \2', text)

  # add extra space to make things easier
  text = " " + text + " "

  # ending quotes
  #text = re.sub(r'"', " '' ", text)
  text = re.sub(r'("|”)', " \1 ", text)

  text = re.sub(r" '([^'])", r" ' \1", text) # ADDED

  #text = re.sub(r'(\S)(\'\')', r'\1 \2 ', text)
  text = re.sub(r'(\S)(\'\')', r'\1 \2 ', text)

  #text = re.sub(r"([^' ])('[sS]|'[mM]|'[dD]|') ", r"\1 \2 ", text)
  #text = re.sub(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ", r"\1 \2 ", text)

  return text.split()


PHONEMAP = json.loads(open('ipatoarpa.json', 'r').read())

def ipa_to_arpabet(phone):
  if phone in PHONEMAP:
    arpa = PHONEMAP.get(phone)
    if is_vowel(arpa):
      return arpa + '0'
    return arpa
  elif phone.startswith(u'ˈ') and phone[1:] in PHONEMAP:
    return PHONEMAP.get(phone[1:]) + '1'
  elif phone.startswith(u'ˌ') and phone[1:] in PHONEMAP:
    return PHONEMAP.get(phone[1:]) + '2'

def alliterates(i1, i2):
  """Test whether the two phones alliterate by Anglo-Saxon standards"""
  # vowels always "alliterate" (initial glottal stop)
  return i1 == i2 or (is_vowel(i1) and is_vowel(i2))

def is_vowel(phoneme):
  if phoneme: return phoneme[0] in 'AEIOU'
  else: return False

def untokenize(tokens):
  return ''.join([' '+t if not (t.startswith("'") or t.startswith('`')) and t not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_``{|}~' else t for t in tokens]).strip()