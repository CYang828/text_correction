# encoding:utf8

import sys
sys.path.append('..')
from nlp import is_realword
del sys


def _make_noisy_channel(word):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in alphabet]
    inserts = [L + c + R for L, R in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def make_noisy_channel(word, distance=1):
    confusion_words = set(word for word in _make_noisy_channel(word) if is_realword(word))
    distance -= 1
    if distance:
        confusion_words_set = set()
        for word in confusion_words:
            confusion_words_set = confusion_words_set.symmetric_difference(make_noisy_channel(word, distance))
        return confusion_words_set
    else:
        return confusion_words


def estimate_distance(word):
    return len(word)//3


if __name__ == '__main__':
    w = 'halle'
    print(make_noisy_channel(w, distance=estimate_distance(w)))
