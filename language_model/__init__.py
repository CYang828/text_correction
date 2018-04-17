# encoding:utf-8

import numpy as np
import tensorflow as tf
from language_model.data import PtbData
from language_model.model import FirstModel


def run_epoch(epochs=10):

    # load data
    d = PtbData('../data/simple-examples/data/ptb.train.txt',
                '../data/simple-examples/data/ptb.valid.txt',
                '../data/simple-examples/data/ptb.test.txt')
    d.produce('train')

    # get data info
    corpus = d.corpus
    corpus_size = corpus.vocab_size
    _, _, batch_len = d.get_batch_data()

    # initialize model & input corpus
    m = FirstModel(corpus, corpus_size)

    # train
    s = tf.Session()
    s.run(tf.global_variables_initializer())
    print('Train start...')
    for epoch in range(epochs):
        for _ in range(batch_len):
            cur_batch, x_batch, y_batch = d.next_batch('train')
            feed_dict = {m.input: x_batch, m.target: y_batch}
            s.run(m.optimizer, feed_dict=feed_dict)

            # train stage information
            if cur_batch % 100 == 0:
                cost = s.run(m.cost, feed_dict=feed_dict)
                print('Epoch: {0:>3}, Batch:{1:>6}, Loss:{2:>6.3}'.format(epoch+1, cur_batch, cost))
                pred = s.run(m.prediction, feed_dict=feed_dict)
                word_ids = s.run(tf.argmax(pred, 1))
                print('Predicted:', ' '.join(corpus.vocab[w] for w in word_ids))
                target_ids = np.argmax(y_batch, 1)
                print('True:', ' '.join(corpus.vocab[w] for w in target_ids))
    print('Train finish.')
    s.close()


if __name__ == '__main__':
    run_epoch()

