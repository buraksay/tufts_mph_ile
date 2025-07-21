from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def porter_stemmer(text):
    """
    Apply Porter stemming to the input text.
    """
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def snowball_stemmer(text):
    """
    Apply Snowball stemming to the input text.
    """
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("english")
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)
