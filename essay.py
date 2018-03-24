# coding:utf-8

from nlp import English, NLP, is_realword


def make_essay(text):
    """将文本转换成文章对象"""
    return Essay(text)


class Essay(object):
    """文章对象"""

    def __init__(self, text):
        self.text = text
        self.paragraphs = self._make_paragraph()
        self._make_nlp()

    def __str__(self):
        return '<Essay>'

    def _make_paragraph(self):
        """生成段落"""
        return [Paragraph(text) for text in self.text.split('\n') if text]

    def _make_nlp(self):
        """"使用nlp能力构建好文章全部结果"""
        for paragraph in self.iter_paragraph():
            documented_paragraph = English(paragraph.text.strip())
            paragraph.nlp = NLP(documented_paragraph)

    def iter_paragraph(self):
        """迭代段落对象"""
        for paragraph in self.paragraphs:
            yield paragraph

    def iter_sentence(self):
        """迭代句子对象"""
        for paragraph in self.iter_paragraph():
            for sentence in paragraph.sentences:
                yield sentence

    def iter_word(self):
        """迭代单词对象"""
        for sentence in self.iter_sentence():
            for word in sentence.words:
                yield word


class Paragraph(object):
    """段落对象"""

    def __init__(self, text):
        self.text = text
        self._nlp = None
        self.sentences = None

    def __str__(self):
        return '<Paragraph>'

    @property
    def nlp(self):
        return self._nlp

    @nlp.setter
    def nlp(self, v):
        self._nlp = v
        self.sentences = self._make_sentence()

    def _make_sentence(self):
        """生成句子"""
        sentence = ''
        sentences = []
        words = []
        for i, token in enumerate(self._nlp.tokens):
            if token.text in ('.', '!', '?'):
                sentence += token.text
                sentences.append(Sentence(sentence, words, self._nlp.tokens))
                sentence = ''
                words = []
            else:
                sentence += token.text + ' '
                words.append(token.text)

        # 如果最后一句没有终止符号
        if sentence:
            sentences.append(Sentence(sentence, words, self._nlp.tokens))
        return sentences


class Sentence(object):
    """句子对象

    简单的语句切分方式为通过一些标点符号来进行，但是这样做会有局限性。
    英文中有些时候'.'并不一定代表是句子的结束，中文中可能也存在这种情况"""

    def __init__(self, text, words, tokens):
        self.text = text
        self.words = [Word(word, token) for word, token in zip(words, tokens)]

    def __str__(self):
        return '<Sentence>'


class Word(object):
    """单词对象"""

    def __init__(self, text, token):
        self.text = text
        self.token = token

    def __str__(self):
        return '<Word>'

    def is_realword(self):
        return is_realword(self.token.lemma_)


if __name__ == '__main__':
    essay = """   English is a internationaly language which becomes importantly for modern world.

    In China, English is took to be a foreigh language which many H.M student choosed to learn. They begin to studying English at a early age. They use at least one hour to learn English knowledges a day. Even kids in kindergarten have begun learning simple words. That's a good phenomenan, for English is essential nowadays.

    In addition to, some people think English is superior than Chinese. In me opinion, though English is for great significance, but English is after all a foreign language. it is hard for people to see eye to eye. English do help us read English original works, but Chinese helps us learn a true China. Only by characters Chinese literature can send off its brilliance. Learning a country's culture, especial its classic culture, the first thing is learn its language. Because of we are Chinese, why do we give up our mother tongue and learn our owne culture through a foreign language?"""

    essay = """good better best chinese"""
    essay = make_essay(essay)
    for w in essay.iter_word():
        print(w)
        if w.is_realword():
            print(w.text, w.token.lemma_)

