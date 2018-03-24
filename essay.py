# coding:utf-8


def make_essay(text):
    """将文本转换成文章对象"""
    return Essay(text)


class Essay(object):
    """文章对象"""

    def __init__(self, text):
        self.text = text
        self.paragraphs = self._make_paragraph()

    def __str__(self):
        return '<Essay>'

    def _make_paragraph(self):
        """生成段落"""

        return [Paragraph(text) for text in self.text.split('\n') if text]


class Paragraph(object):
    """段落对象"""

    def __init__(self, text):
        self.text = text

    def __str__(self):
        return '<Paragraph>'


if __name__ == '__main__':
    essay = """   English is a internationaly language which becomes importantly for modern world.

    In China, English is took to be a foreigh language which many student choosed to learn. They begin to studying English at a early age. They use at least one hour to learn English knowledges a day. Even kids in kindergarten have begun learning simple words. That's a good phenomenan, for English is essential nowadays.

    In addition to, some people think English is superior than Chinese. In me opinion, though English is for great significance, but English is after all a foreign language. it is hard for people to see eye to eye. English do help us read English original works, but Chinese helps us learn a true China. Only by characters Chinese literature can send off its brilliance. Learning a country's culture, especial its classic culture, the first thing is learn its language. Because of we are Chinese, why do we give up our mother tongue and learn our owne culture through a foreign language?"""
    print(make_essay(essay).paragraphs)

