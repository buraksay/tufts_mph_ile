import itertools

import pandas as pd

# Define the long-form options for each dimension
stopwords = ['Standard Stopwords', 'Custom Stopwords', 'Remove Punctuation']
stemmer = ['Porter Stemmer', 'Alternative Stemmer']
ngram = ['Uni-gram', 'Bi-gram', 'Tri-gram']
vectorizer = ['TF-IDF', 'BoW']
classifier = ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'XGBoost']

# Define mappings from long-form to abbreviation
stopwords_map = {'Standard Stopwords': 'std', 'Custom Stopwords': 'csw', 'Remove Punctuation': 'pun'}
stemmer_map = {'Porter Stemmer': 'por', 'Alternative Stemmer': 'alt'}
ngram_map = {'Uni-gram': 'uni', 'Bi-gram': 'bi', 'Tri-gram': 'tri'}
vectorizer_map = {'TF-IDF': 'tfidf', 'BoW': 'bow'}
classifier_map = {'Logistic Regression': 'lr', 'Naive Bayes': 'nb', 'Random Forest': 'rf', 'XGBoost': 'xgb'}

combinations = list(itertools.product(stopwords, stemmer, ngram, vectorizer, classifier))

# Create DataFrame with ID and Tag
df = pd.DataFrame(combinations, columns=['Stopwords', 'Stemmer', 'N-gram', 'Vectorizer', 'Classifier'])
df['ID'] = ['EXP{:03d}'.format(i+1) for i in range(len(df))]
df['Tag'] = df.apply(lambda row: f"{stopwords_map[row['Stopwords']]}-{stemmer_map[row['Stemmer']]}-{ngram_map[row['N-gram']]}-{vectorizer_map[row['Vectorizer']]}-{classifier_map[row['Classifier']]}", axis=1)

# Rearrange columns for readability
df = df[['ID', 'Tag', 'Stopwords', 'Stemmer', 'N-gram', 'Vectorizer', 'Classifier']]

# Save for later use
df.to_csv('nlp_experiment_matrix.csv', index=False)
print(df.head())
