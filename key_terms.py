from lxml import etree
import nltk
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    xml_path = "news.xml"
    corpus = etree.parse(xml_path).getroot()[0]
    wnl = WordNetLemmatizer()
    vectorizer = TfidfVectorizer()
    ignore = list(string.punctuation) + stopwords.words('english') + ['ha', 'wa', 'u', 'a']
    all_documents = []
    for new_story in corpus:
        head, story = new_story
        text = story.text
        tokens = nltk.tokenize.word_tokenize(text.lower())
        tokens.sort(reverse=True)
        norm = [wnl.lemmatize(term) for term in tokens]
        norm_filtered = [nltk.pos_tag([x])[0] for x in norm if x not in ignore]
        post_tagged = [x[0] for x in norm_filtered if x[1] == "NN"]
        all_documents.append(" ".join(post_tagged))
    tfidf_matrix = vectorizer.fit_transform(all_documents)
    terms = vectorizer.get_feature_names()
    for index, document in enumerate(tfidf_matrix.toarray()):
        ordered_document = []
        for index2, value in enumerate(document):
            if value:
                ordered_document.append((index2, value, terms[index2]))
        ordered_document.sort(key=lambda x: (x[1], x[2]), reverse=True)
        five_index = ordered_document[:5]
        ordered_terms = []
        for item in five_index:
            ordered_terms.append(terms[item[0]])
        head, _ = corpus[index]
        print(head.text, end=":\n")
        print(*ordered_terms, sep=' ', end="\n\n")



if __name__ == '__main__':
    main()
