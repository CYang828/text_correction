# encoding:utf-8

import numpy as np
import tensorflow as tf
from data import PtbData
from model import FirstModel


def run_epoch(epochs=10):

    data = PtbData('../data/simple-examples/data/ptb.train.txt',
                   '../data/simple-examples/data/ptb.valid.txt',
                   '../data/simple-examples/data/ptb.test.txt')
    data.producer('train')
    corpus, corpus_size, _, batch_len = data.get('train')
    m = FirstModel(corpus)

    s = tf.Session()
    s.run(tf.global_variables_initializer())

    print('训练开始...')
    for epoch in range(epochs):
        for _ in range(batch_len):
            cur_batch, x_batch, y_batch = data.next_batch('train')
            feed_dict = {m.input: x_batch, m.target: y_batch}
            s.run(m.optimizer, feed_dict=feed_dict)

            if cur_batch % 100 == 0:
                cost = s.run(m.cost, feed_dict=feed_dict)
                print('Epoch: {0:>3}, Batch:{1:>6}, Loss:{2:>6.3}'.format(epoch+1, cur_batch, cost))

                pred = s.run(m._prediction, feed_dict=feed_dict)
                word_ids = s.run(tf.argmax(pred, 1))
                print('Predicted:', ' '.join(corpus.vocab[w] for w in word_ids))
                target_ids = np.argmax(y_batch, 1)
                print('True:', ' '.join(corpus.vocab[w] for w in target_ids))

    print('结束训练...')
    s.close()


if __name__ == '__main__':
    run_epoch()

