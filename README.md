# tufts_mph_ile

## How to run
* create conda environment nlp4ori
* run ```python experiments/create_experiments_matrix.py```  to generate the input file with list of experiment combinations
* ...
* ... *describe how to run a batch for the whole set of experiments*
* ...

a single_task (which is a combination of stopword, stemmer, ngram, and classifier) is invoked like so:
```
CODE_DIR="/Users/baba/proj/aphaproject/tufts_mph_ile/" python single_task.py \
--id "EXP002" \
--tag "sw.std-stem.snow-ngram.tri-vec.tfidf-clf.lr" \
--batch "batch_202507220123"
```

note batch param is optional. if not provided, system time will be used. 

# TODO

- [ ] Create custom stopwords
- [ ] explore min_df max_df

- [x] implement snowball stemmer
- [ ] explore punctuation cleaner
- [x] implement bag of words
- [x] implement tfidf
- [x] implement bigram and trigram

- [x] implement logistic regression
- [ ] implement random forest
- [ ] implement naïve bayes
- [ ] implement xgboost
- [ ] implement task list parser
- [ ] implement bash runner

- [ ] save roc-aux after each run
- [ ] save confusion matrix after each run

