# encoding:utf-8

import numpy as np
from language_model.corpus import Corpus


class PtbData(object):

    def __init__(self, train_file, valid_file, test_file):
        self.train_file = train_file
        self.valid_file = valid_file
        self.text_file = test_file

        # load train set corpus
        self.corpus = Corpus(train_file).load()

        # train set convert to index
        self._train = self.corpus.text2index(train_file)
        # valid set convert to index
        self._valid = self.corpus.text2index(valid_file)
        # test set convert to index
        self._test = self.corpus.text2index(test_file)

        # index convert data
        self._index_data = {'train': (self._train, len(self._train)),
                            'valid': (self._valid, len(self._valid)),
                            'test': (self._test, len(self._test))}

        # batch convert data
        self._batch_data = {}

        # current batch
        self._cur_batch = {'train': 0,
                           'valid': 0,
                           'test': 0}

    def get_index_data(self, setype='train'):
        return self._index_data.get(setype)

    def get_batch_data(self, setype='train'):
        return self._batch_data.get(setype)

    def produce(self, setype='train', batch_size=64, step=20, stride=1):
        """:returns:
            - (sentence, nextword, batch_len)
        """

        data_set, data_len = self.get_index_data(setype)

        # generate sentence
        sentence = []
        # nextword is the target word of the sentence
        nextword = []

        for i in range(0, data_len - step, stride):
            sentence.append(self._train[i: (i+step)])
            nextword.append(self._train[i+step])
        sentence = np.array(sentence)
        nextword = np.array(nextword)
        batch_len = len(sentence) // batch_size

        x = np.reshape(sentence[:(batch_len*batch_size)], [batch_len, batch_size, -1])
        y = np.reshape(nextword[:(batch_len*batch_size)], [batch_len, batch_size])
        self._batch_data[setype] = (x, y, batch_len)
        return self

    def next_batch(self, setype='train'):
        x, y, batch_len = self._batch_data.get(setype)
        corpus_size = self.corpus.vocab_size
        cur_batch = self._cur_batch.get(setype, 0)
        y = y[cur_batch]

        # index to one-hot
        y_ = np.zeros((y.shape[0], corpus_size), dtype=np.bool)
        for i in range(y.shape[0]):
            y_[i][y[i]] = 1
        self._cur_batch[setype] = (cur_batch + 1) % batch_len
        return cur_batch, x[cur_batch], y_


if __name__ == '__main__':
    data = PtbData('../data/simple-examples/data/ptb.train.txt',
                   '../data/simple-examples/data/ptb.valid.txt',
                   '../data/simple-examples/data/ptb.test.txt').produce()
    print(data.next_batch())
    print(data.next_batch())
    print(data.next_batch())




