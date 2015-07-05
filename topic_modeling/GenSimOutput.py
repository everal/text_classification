#coding:utf-8
import logging
import jieba
from gensim import corpora, models, similarities

import myConfig
import preprocess

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#自定义需要查找与目标最相似的文档
target_terms = u'大神帮忙忙,软件刷机升级黑屏方法'


documents = preprocess.get_docs(myConfig.contents_seged_file)

dictionary = corpora.Dictionary.load(myConfig.dict_file)
corpus = corpora.MmCorpus(myConfig.corpora_file)

model_transform = models.LsiModel.load(myConfig.topic_model_file)


####3
target_terms = target_terms.encode('utf-8')

jieba.load_userdict(myConfig.user_dict_file)
target_terms = ' '.join( jieba.cut(target_terms.strip() , cut_all = False) ).encode('utf-8')
vec_bow = dictionary.doc2bow(target_terms.split())
vec_lsi = model_transform[vec_bow] # convert the query to model(like LDA LSI) space
print(vec_lsi)

logging.info('bigin MatrixSimilarity...')
index = similarities.MatrixSimilarity(model_transform[corpus]) # transform corpus to model(like LDA LSI) space and index it
logging.info('finish MatrixSimilarity...')

index.save(myConfig.similarity_index_file)

# index = similarities.MatrixSimilarity.load(''tech_forum_model.index')
sims = index[vec_lsi]
print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])

index = 0
fout = open(myConfig.similarity_output_file, 'w')
for item in sims:
    index += 1
    result = str(index) + ": " + documents[item[0]] + '\t'+ str(item[1])
    fout.write(result +'\n')
fout.close()