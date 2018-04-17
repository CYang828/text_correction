# encoding:utf-8

import tensorflow as tf


class FirstModel(object):
    def __init__(self, corpus, corpus_size, batch_size=64, step=20, stride=3,
                 embedding_dim=64, hidden_dim=128, layer=2, learning_rage=0.05, dropout=0.2, rnn_model='gru'):
        # 语料库
        self.corpus = corpus
        self.corpus_size = corpus_size
        # 每一批数据大小
        self.batch_size = batch_size
        # 句子长度
        self.step = step
        # 取数据步长
        self.stride = stride
        # 词向量纬度
        self.embedding_dim=embedding_dim
        # RNN隐藏层纬度
        self.hidden_dim = hidden_dim
        # 隐藏层层数
        self.layer = layer
        # 学习速率
        self.learning_rate = learning_rage
        # 每个神经元被丢弃的概率
        self.dropout = dropout
        # RNN神经单元类型
        self.rnn_model = rnn_model

        # 输入输出占位符
        self.input = tf.placeholder(tf.int32, [None, self.step])
        self.target = tf.placeholder(tf.int32, [None, self.corpus_size])

        self.logistic, self.prediction = self.rnn()
        self.cost = self.cost()
        self.optimizer = self.optimizer()
        self.error = self.error()

    def input_embedding(self):
        with tf.device("/device:GPU:0"):
            embedding = tf.get_variable('embedding', [self.corpus_size, self.embedding_dim], dtype=tf.float32)
            _input = tf.nn.embedding_lookup(embedding, self.input)
        return _input

    def rnn(self):
        def lstm_cell():
            """lstm神经元"""
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)

        def gru_cell():
            """gru神经元"""
            return tf.contrib.rnn.GRUCell(self.hidden_dim)

        def dropout_cell():
            """dropout包装"""
            if self.rnn_model == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()

            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)

        cells = [dropout_cell() for _ in range(self.layer)]
        cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        _input = self.input_embedding()
        _output, state = tf.nn.dynamic_rnn(cell=cells, inputs=_input, dtype=tf.float32)

        last = _output[:, -1, :]
        logistic = tf.layers.dense(inputs=last, units=self.corpus_size)
        prediction = tf.nn.softmax(logistic)
        return logistic, prediction

    def cost(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logistic, labels=self.target)
        cost = tf.reduce_mean(cross_entropy)
        return cost

    def optimizer(self):
        optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimize.minimize(self.cost)

    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
