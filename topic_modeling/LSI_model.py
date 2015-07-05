#coding:utf-8
import logging
import jieba
from gensim import corpora, models
import myConfig
import preprocess

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dictionary = corpora.Dictionary.load(myConfig.dict_file)
corpus = corpora.MmCorpus(myConfig.corpora_file)

if myConfig.useTFIDF:
    tfidf = models.TfidfModel(corpus)
    corpus = tfidf[corpus]

lsi = models.LsiModel(corpus, id2word =dictionary, num_topics = myConfig.num_topics )
corpus_lsi = lsi[corpus]
lsi.save(myConfig.topic_model_file)
lsi.print_topics(myConfig.num_topics)
lsi.print_debug(myConfig.num_topics, num_words = 20 )


