import logging

LOG_LEVEL = logging.INFO
LOG_MSG_FORMAT= "[ %(asctime)s %(levelname)s - %(funcName)20s() ] %(message)s"
LOG_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

N_SPLITS = 5
SCORING = "roc_auc"
DATA_FILE_NAME = "Data_combined_122024.xlsx"
RES_DIR_NAME = "results"
DATA_DIR = "/Users/baba/proj/aphaproject/burak_local/data"
SEED = 202507
SPLIT_DATE = "2020"

# Algorithm mapping constants
SW_STD = "sw.std"
SW_CSW = "sw.csw"
SW_PUN = "sw.pun"
STEM_POR = "stem.por"
STEM_SNOW = "stem.snow"
NGRAM_UNI = "ngram.uni"
NGRAM_BI = "ngram.bi"
NGRAM_TRI = "ngram.tri"
VEC_TFIDF = "vec.tfidf"
VEC_BOW = "vec.bow"
CLF_LR = "clf.lr"
CLF_NB = "clf.nb"
CLF_RF = "clf.rf"
CLF_XGB = "clf.xgb"

# Function name constants
STANDARD_STOPWORDS = "standard_stopwords"
CUSTOM_STOPWORDS = "custom_stopwords"
REMOVE_PUNCTUATION = "remove_punctuation"
PORTER_STEMMER = "porter_stemmer"
SNOWBALL_STEMMER = "snowball_stemmer"
UNIGRAM = (1, 1)
BIGRAM = (1, 2)
TRIGRAM = (1, 3)
TFIDF = "TfidfVectorizer"
BAG_OF_WORDS = "CountVectorizer"
LOGISTIC_REGRESSION = "logistic_regression"
NAIVE_BAYES = "naive_bayes"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"

#performance metrics
F1_SCORE = "f1_score"
ROC_AUC = "roc_auc"
PRECISION = "precision"
RECALL = "recall"
ACCURACY = "accuracy"
