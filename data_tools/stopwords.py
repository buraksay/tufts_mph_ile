from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STANDARD_SW = set(stopwords.words('english'))
CUSTOM_SW = set(["example", "custom", "stopword", "list"])

def standard_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in STANDARD_SW]
    return ' '.join(tokens)

def custom_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in CUSTOM_SW]
    return ' '.join(tokens)

# noop
def remove_punctuation(text):
    return text
