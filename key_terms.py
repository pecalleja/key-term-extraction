from lxml import etree
import nltk
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string


def main():
    xml_path = "news.xml"
    corpus = etree.parse(xml_path).getroot()[0]
    wnl = WordNetLemmatizer()
    ignore = list(string.punctuation) + stopwords.words('english')
    for new_story in corpus:
        head, story = new_story
        text = story.text
        tokens = nltk.tokenize.word_tokenize(text.lower())
        tokens.sort(reverse=True)
        norm = [wnl.lemmatize(term) for term in tokens]
        norm_filtered = [nltk.pos_tag([x])[0] for x in norm if x not in ignore]
        post_tagged = [x[0] for x in norm_filtered if x[1] == "NN"]
        freq_list = Counter(post_tagged).most_common(5)
        print(head.text, end=":\n")
        freq_words = [x[0] for x in freq_list]
        print(*freq_words, sep=' ', end="\n\n")


if __name__ == '__main__':
    main()
