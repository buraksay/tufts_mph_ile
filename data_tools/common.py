import re

'''
convert to lowercase
remove punctuation, digits, URLs etc.
tokenize
stopword removal
stemming/lemmatization
vectorize + ngramming (combine and then generate ngrams)
'''

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#","", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text
