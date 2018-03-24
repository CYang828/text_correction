# encoding:utf-8

import en_core_web_sm


def load_model():
    return en_core_web_sm.load()


def is_realword(word):
    return word in English.vocab


class NLP(object):
    """nlp中间件"""

    def __init__(self, nlp_object):
        self.tokens = nlp_object


English = load_model()
