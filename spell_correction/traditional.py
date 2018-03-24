# encoding:utf-8

import spacy
import en_core_web_sm

English = en_core_web_sm.load()

doc = English('Apple is looking at buying U.K.')

for token in doc:
    print(token)


class NonWordCorrection(object):

    def detection(self):
        pass

    def correction(self):
        pass


class RealWordCorrection(object):

    def detection(self):
        pass

    def correction(self):
        pass


class SpellCorrection:

    @staticmethod
    def correction(original_text):
        doc = English(original_text)
        print(dir(doc))
        print([s for s in doc.sents])
        pass


if __name__ == '__main__':
    SpellCorrection.correction("hallo, my name is zhang. I'm 25 years old")




