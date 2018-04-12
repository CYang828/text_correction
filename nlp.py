# encoding:utf-8

import en_core_web_sm
#import en_core_web_md
import en_core_web_lg


def load_model():
    return en_core_web_lg.load()


def is_realword(word):
    return word in English.vocab


class NLP(object):
    """nlp中间件"""

    def __init__(self, nlp_object):
        self.tokens = nlp_object


English = load_model()


if __name__ == '__main__':
    load_model()
    print(is_realword('especial'))


