#coding=utf-8
#!usr/bin/python
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import ShuffleSplit,train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score,fbeta_score
from sklearn.base import BaseEstimator

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
import datetime
import codecs
import csv
import logging
import jieba
from gensim import corpora, models, similarities
import myConfig
import preprocess

starttime = datetime.datetime.now()
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
try:
    import cPickle as pickle
except:
    import pickle

# dictionary = corpora.Dictionary.load('tech_forum-31.dict')
# lsi = models.LsiModel.load('tech_forum_lsimodel-31.lsi')
#
def LSI_transform(content):
    vec_bow = dictionary.doc2bow(content.split())
    vec_lsi = lsi[vec_bow]
    vec_lsi = [topic_val[1] for topic_val in vec_lsi  ]
    return vec_lsi



dictionary = corpora.Dictionary.load(myConfig.dict_file)
lda        = models.LdaModel.load(myConfig.topic_model_file)

def LDA_transform(content):
    vec_bow = dictionary.doc2bow(content.split())
    vec_lda = lda[vec_bow]
    res = [-1]*10
    for value in vec_lda:
        res[value[0]] = value[1]
    return res

def read_taged_data(train_path):
    # X = [[0.1,0.1], [0.2, 0.2], [0.3, 0.3], [-0.1, -0.1], [-0.2, -0.2], [-0.3, -0.3] ,[0.12,0.12]]
    # Y = ['1', '1', '1', '0', '0', '0', '0']
    X = []
    Y = []
    for line in codecs.open(train_path):
        line = line.encode('utf-8').strip()
        tag = line[-1]
        content = line[:-1].strip('\'",')
        X.append(LDA_transform(' '.join(content.split())))
        Y.append( tag.strip() )

    return X, Y

def my_custom_f1_score(ground_truth, predictions):
        score = f1_score( ground_truth, predictions, pos_label='1')
        # score = fbeta_score(ground_truth, predictions, pos_label='1', beta = 0.3)
        # beta=1 相当于f1_score
        # F2分数，准确率的权重高于召回率，
        # F0.5，召回率的权重高于准确率。
        #
        return score


def gridSearch(train_path, model_path):

    train, label = read_taged_data(train_path)
    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.1, random_state=1)

    print 'length of training data : ' + str(len(y_train))

    pipeline = Pipeline([
                        # ('selection' , SelectKBest(chi2, k=200 )),
                         ('Random_forest', RandomForestClassifier(n_estimators=100))])

    cv = ShuffleSplit(
        n=len(X_train), n_iter=4, test_size=0.3, random_state = 1)

    param_grid = dict(
                      Random_forest__n_estimators=[50, 70, 200],
                      )

    my_custom_scorer = make_scorer(my_custom_f1_score, greater_is_better=True)

    grid_search = GridSearchCV(pipeline,
                               param_grid=param_grid,
                               cv=cv,
                               n_jobs = 1,
                               # scoring = my_custom_scorer,
                               verbose=10)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_

    #save classifier object
    try:
        fp = open(model_path,'wb')
    except IOError:
        print 'could not open file:',model_path

    # pickle.dump(grid_search, fp)
    pickle.dump(grid_search.best_estimator_, fp)
    # pickle.dump(grid_search.best_params_, fp)
    # pickle.dump(grid_search.grid_scores_, fp)
    # pickle.dump(tfidf_ngrams, fp)
    # pickle.dump(pipeline, fp)
    fp.close()



    print 'best_score_:', grid_search.best_score_
    print 'best_params_:', grid_search.best_params_

    clf = grid_search.best_estimator_

    predicted = clf.predict(X_test)
    target_names = ['0', '1']
    # target_names = [0, 1]
    print classification_report(y_test, predicted, target_names=target_names)
    print 'time usered:'
    print (datetime.datetime.now() - starttime).seconds



if __name__ == '__main__':

    gridSearch(myConfig.train_file, myConfig.gs_model_file)

