
import logging


def tfidf(text, ngram_range=(1, 1)):
    from sklearn.feature_extraction.text import TfidfVectorizer
    logging.info("Using TF-IDF vectorizer with ngram range: %s", ngram_range)
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform([text])
    return tfidf_matrix

def bag_of_words(text, ngram_range=(1, 1)):
    from sklearn.feature_extraction.text import CountVectorizer
    logging.info("Using Bag of Words vectorizer with ngram range: %s", ngram_range)
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    bow_matrix = vectorizer.fit_transform([text])
    return bow_matrix
