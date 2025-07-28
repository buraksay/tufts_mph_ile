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

- [x] Create custom stopwords
- [x] explore min_df max_df

- [x] implement snowball stemmer
- [x] explore punctuation cleaner
- [x] implement bag of words
- [x] implement tfidf
- [x] implement bigram and trigram

- [x] implement logistic regression
- [x] implement random forest
- [x] implement naïve bayes
- [x] implement xgboost
- [x] implement task list parser
- [x] implement bash runner

- [x] save roc-aux after each run
- [ ] save confusion matrix after each run

- [ ] design a central cache the preprocessed data in each step to reuse with subsequent steps (would shave around 2 hours from the total run, as the clean-stopwords-stemming takes ~60 seconds - would run only first 6 variations, then retrieved from the cache for the subsequent runs)

```
[ 07/27/2025 22:53:57 INFO -      run_single_task() ] Calling data_tools.read_xlsx_into_x_y_k_dfs
[ 07/27/2025 22:53:57 INFO -      run_single_task() ] INPUT xlsx_path: /cluster/tufts/shresthaapha/bsay01/data/Data_combined_122024.xlsx
[ 07/27/2025 22:54:05 INFO -      run_single_task() ] Running task EXP004 with tag sw.std-stem.por-ngram.bi-vec.bow-clf.rf
[ 07/27/2025 22:54:05 INFO -      run_single_task() ] Output directory: /cluster/tufts/shresthaapha/bsay01/tufts_mph_ile/results/batch_202507272243/EXP004_sw.std-stem.por-ngram.bi-vec.bow-clf.rf
[ 07/27/2025 22:54:05 INFO -      preprocess_data() ] Preprocessing data for task EXP004 with tag sw.std-stem.por-ngram.bi-vec.bow-clf.rf and algorithms ['sw.std', 'stem.por', 'ngram.bi', 'vec.bow', 'clf.rf']
[ 07/27/2025 22:54:05 INFO -      preprocess_data() ] applying clean
[ 07/27/2025 22:54:08 INFO -        save_csv_file() ] saved x_df_clean.csv with shape (15220, 1) to file /cluster/tufts/shresthaapha/bsay01/tufts_mph_ile/results/batch_202507272243/EXP004_sw.std-stem.por-ngram.bi-vec.bow-clf.rf/x_df_clean.csv
[ 07/27/2025 22:54:08 INFO -    get_stopword_func() ] Using standard stopwords function
[ 07/27/2025 22:54:08 INFO -      preprocess_data() ] applying stopwords with function: standard_stopwords
[ 07/27/2025 22:54:16 INFO -        save_csv_file() ] saved x_df_stopwords.csv with shape (15220, 1) to file /cluster/tufts/shresthaapha/bsay01/tufts_mph_ile/results/batch_202507272243/EXP004_sw.std-stem.por-ngram.bi-vec.bow-clf.rf/x_df_stopwords.csv
[ 07/27/2025 22:54:16 INFO -     get_stemmer_func() ] Using Porter stemmer
[ 07/27/2025 22:54:16 INFO -      preprocess_data() ] applying stemmer with function: porter_stemmer
[ 07/27/2025 22:54:55 INFO -        save_csv_file() ] saved x_df_stemmer.csv with shape (15220, 1) to file /cluster/tufts/shresthaapha/bsay01/tufts_mph_ile/results/batch_202507272243/EXP004_sw.std-stem.por-ngram.bi-vec.bow-clf.rf/x_df_stemmer.csv
[ 07/27/2025 22:54:55 INFO -           split_data() ] Splitting data into train and test sets with split date 2020 and seed 202507
[ 07/27/2025 22:54:55 INFO - split_into_train_test() ] Train size: 10373, Test size: 4847
[ 07/27/2025 22:54:55 INFO - split_into_train_test() ] Train date range: 2011-01-01 00:00:00 to 2019-12-31 00:00:00
[ 07/27/2025 22:54:55 INFO - split_into_train_test() ] Test date range: 2020-01-01 00:00:00 to 2024-10-30 00:00:00
```
