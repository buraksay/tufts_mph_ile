# You need to install it and download a language model first
# pip install pycontractions
# python -m spacy download en_core_web_sm

import gensim.downloader as api
from pycontractions import Contractions

# Load a pre-trained model (e.g., GloVe)
model = api.load("glove-twitter-25")

cont = Contractions(kv_model=model)
cont.load_models()

text = "I've been thinking, but I can't decide. It's a tough choice."

# The expand_texts method returns a list of expanded texts
expanded_text = list(cont.expand_texts([text], precise=True))[0]

print(f"Original: {text}")
print(f"Expanded: {expanded_text}")
# Expected Output:
# Original: I've been thinking, but I can't decide. It's a tough choice.
# Expanded: I have been thinking, but I cannot decide. It is a tough choice.
