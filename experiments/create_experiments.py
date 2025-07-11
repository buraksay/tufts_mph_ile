import itertools

import pandas as pd

stopwords = ["Standard", "Custom", "Punctuation"]
stemmer = ["Porter", "Alternative"]
ngram = ["Uni-gram", "Bi-gram", "Tri-gram"]
vectorizer = ["TF-IDF", "BoW"]
classifier = ["Logistic Regression", "Naive Bayes", "Random Forest", "XGBoost"]

combinations = list(
    itertools.product(stopwords, stemmer, ngram, vectorizer, classifier)
)
df = pd.DataFrame(
    combinations, columns=["Stopwords", "Stemmer", "N-gram", "Vectorizer", "Classifier"]
)
df.to_csv("nlp_experiments.csv", index=False)
print("Experiment combinations saved to nlp_experiments.csv")
# This code generates all combinations of the specified parameters for NLP experiments
# and saves them to a CSV file named "nlp_experiments.csv". Each row in the CSV represents a unique combination of parameters:
# - Stopwords: Standard or Custom
# - Stemmer: Porter or Alternative
# - N-gram: Uni-gram, Bi-gram, or Tri-gram
# - Vectorizer: TF-IDF or BoW
# - Classifier: Logistic Regression, Naive Bayes, Random Forest, or XGBoost 
