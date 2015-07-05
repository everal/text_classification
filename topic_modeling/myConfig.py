#coding=utf-8
#!usr/bin/python
import sys

reload(sys)
sys.setdefaultencoding('utf8')

#[gloabal]
#停用词列表
stopword_file = 'exdata/stopword.txt'

#自定义字典
user_dict_file = 'exdata/user.dict'

#分词后的文件
contents_seged_file = 'exdata/content-seged.data'
#gensim 生成的字典文件
dict_file = 'exdata/tech_forum.dict'

#gensim 生成的语料文件
corpora_file = 'exdata/tech_forum.mm'

#gensim 生成的topic model文件
topic_model_file = 'exdata/tech_forum_model.lsi'

#gensim 相似度计算
similarity_output_file = 'exdata/tech_sim_result.data'
similarity_index_file = 'exdata/tech_forum_model.index'

useTFIDF = True
num_topics  = 10

#[grid_search]
train_file       = u'exData/tagData.csv'
gs_model_file    = u'exData/17-test.grids.model'