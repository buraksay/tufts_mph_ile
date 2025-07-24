import re

'''
convert to lowercase
remove punctuation, digits, URLs etc.
tokenize
stopword removal
stemming/lemmatization
vectorize + ngramming (combine and then generate ngrams)
'''

X000D = "_x000d_"
def clean(narrative):
    narrative = narrative.lower()
    narrative = re.sub(r"_x000d_|x000d", "", narrative)
    narrative = re.sub(r"http\S+|www\S+|https\S+", "", narrative, flags=re.MULTILINE)
    # Remove extra whitespace and newlines
    narrative = re.sub(r"\s+", " ", narrative).strip()
    # matches @mentions and hashtags
    # narrative = re.sub(r"\@\w+|\#","", narrative)
    # narrative = re.sub(r"[^a-zA-Z0-9\s]", "", narrative)
    narrative = re.sub(r"[^a-zA-Z\s]", "", narrative)
    return narrative
