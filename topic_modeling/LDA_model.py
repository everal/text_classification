#coding:utf-8
import logging
from gensim import corpora, models, similarities
import gensim.models.ldamodel
import codecs
import jieba
import sys
import myConfig
import preprocess
reload(sys)
sys.setdefaultencoding('utf-8')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

texts = preprocess.get_docs(myConfig.contents_seged_file)
texts = preprocess.cleanStopword(texts)

dictionary = corpora.Dictionary.load(myConfig.dict_file)
corpus = corpora.MmCorpus(myConfig.corpora_file)

if myConfig.useTFIDF:
    tfidf = models.TfidfModel(corpus)
    corpus = tfidf[corpus]

lda = models.LdaModel(corpus, id2word = dictionary, num_topics = myConfig.num_topics)
lda.save(myConfig.topic_model_file)
lda.show_topics(myConfig.num_topics, num_words = 10, log = False, formatted = True)

# lda.print_topics(num_topics=10, num_words=20)
# lda.show_topics(num_topics=10, num_words=10, log=False, formatted=True)
# lda.top_topics(corpus, num_topics=10, num_words=20)
