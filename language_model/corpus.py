# encoding:utf-8


from collections import Counter


class Corpus(object):

    def __init__(self, filename):
        self.filename = filename
        self.vocab = None
        self._vocab_index = None
        self.vocab_size = 0

    @staticmethod
    def _read_sentence(sentence):
        return sentence.replace('\n', '<eos>').split()

    @staticmethod
    def _read(filename):
        with open(filename, 'r') as f:
            return Corpus._read_sentence(f.read())

    def load(self):
        data = self._read(self.filename)
        self.vocab = Counter(data)
        self.vocab = sorted(self.vocab.items(), key=lambda x: -x[1])
        self.vocab, _ = list(zip(*self.vocab))
        self.vocab_size = len(self.vocab)
        self._vocab_index = dict(zip(self.vocab, range(self.vocab_size)))
        return self

    def text2index(self, filename=None, sentence=None):
        if filename:
            data = Corpus._read(filename)
        elif sentence:
            data = self._read_sentence(sentence)

        return [self._vocab_index[x] for x in data if x in self._vocab_index]

    def index2text(self, index_sequence):
        return ' '.join(list(map(lambda x: self.vocab[x], index_sequence)))


if __name__ == '__main__':
    c = Corpus('../data/simple-examples/data/ptb.train.txt')
    c.load()
    print(c.text2index(sentence='i love you'))
    print(c.index2text([68, 3248, 110]))


