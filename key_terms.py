from lxml import etree
import nltk
from collections import Counter

xml_path = "news.xml"
corpus = etree.parse(xml_path).getroot()[0]
for new_story in corpus:
    head, story = new_story
    text = story.text
    tokens = nltk.tokenize.word_tokenize(text.lower())
    tokens.sort(reverse=True)
    freq_list = Counter(tokens).most_common(5)
    print(f"{head.text}:")
    freq_words = [x[0] for x in freq_list]
    print(" ".join(freq_words))
    print()
