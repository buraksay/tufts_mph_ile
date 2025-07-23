import nltk
from nltk.stem.snowball import SnowballStemmer

# the stemmer requires a language parameter
snow_stemmer = SnowballStemmer(language="english")

# list of tokenized words
words = [
    "cared",
    "university",
    "fairly",
    "easily",
    "singing",
    "sings",
    "sung",
    "singer",
    "sportingly",
]

# stem's of each word
stem_words = []
for w in words:
    x = snow_stemmer.stem(w)
    stem_words.append(x)

# print stemming results
for e1, e2 in zip(words, stem_words):
    print(e1 + " ----> " + e2)
