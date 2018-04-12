# encoding:utf-8

import numpy as np
from corpus import Corpus


class PtbData(object):

    def __init__(self, train_file, valid_file, test_file):
        self.train_file = train_file
        self.valid_file = valid_file
        self.text_file = test_file
        self._corpus = Corpus(train_file).load()
        # 训练集
        self._train = self._corpus.text2index(train_file)
        # 验证集
        self._valid = self._corpus.text2index(valid_file)
        # 测试集
        self._test = self._corpus.text2index(test_file)
        # 转换后的数据集
        self.dtype_dict = {'train': [self._train, len(self._train), 0, self._train[0]],
                           'valid': [self._valid, len(self._valid), 0, self._valid[0]],
                           'test': [self._test, len(self._test), 0, self._test[0]]}
        # batch后的数据
        self.producer_set = {}

    def producer(self, dtype='train', batch_size=64, step=20, stride=1):
        data_set, data_len, _, _ = self.dtype_dict.get(dtype, 'train')

        sentence = []
        nextword = []

        for i in range(0, data_len - step, stride):
            sentence.append(self._train[i: (i+step)])
            nextword.append(self._train[i+step])

        sentence = np.array(sentence)
        nextword = np.array(nextword)
        batch_len = len(sentence) // batch_size

        x = np.reshape(sentence[:(batch_len*batch_size)], [batch_len, batch_size, -1])
        y = np.reshape(nextword[:(batch_len*batch_size)], [batch_len, batch_size])
        self.producer_set[dtype] = (x, y)
        return self

    def get(self, dtype='train'):
        return self.dtype_dict.get(dtype)

    def next_batch(self, dtype='train'):
        _, corpus_size, cur_batch, batch_len = self.dtype_dict.get(dtype)
        x, y = self.producer_set.get(dtype)

        # 构建one-hot
        y_ = np.zeros((y.shape[0], corpus_size), dtype=np.bool)
        for i in range(y.shape[0]):
            y_[i][y[i]] = 1
        self.dtype_dict.get(dtype)[2] = (cur_batch + 1) % batch_len
        return cur_batch, x[cur_batch], y_


if __name__ == '__main__':
    data = PtbData('../data/simple-examples/data/ptb.train.txt',
                   '../data/simple-examples/data/ptb.valid.txt',
                   '../data/simple-examples/data/ptb.test.txt').producer()
    print(data.next_batch())
    print(data.next_batch())
    print(data.next_batch())




