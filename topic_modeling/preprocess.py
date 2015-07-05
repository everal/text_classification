#coding:utf-8
import json
import codecs
import re
import jieba
import logging
from gensim import corpora
import myConfig

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_docs(fileName):
        documents  = []
        rowcount = 0
        for line in open(fileName):
           cut_result = ' '.join( jieba.cut(line.strip(), cut_all = False) ).encode('utf-8')
           documents.append(cut_result)
           rowcount += 1
        logging.info('finish load %s documents' %(rowcount))
        return documents

def get_stopword_list(fileName):
    list  = []
    for word in codecs.open(fileName):
        list.append(word.encode('utf-8').strip())
    return set(list)


jieba.load_userdict(myConfig.user_dict_file)

def seg(sentence, cleanStopword = True):
    stoplist = get_stopword_list(myConfig.stopword_file)
    sentence = ' '.join( jieba.cut(sentence , cut_all = False, HMM = True) ).encode('utf-8')
    if cleanStopword:
        sentence = ' '.join([word for word in sentence.split() if word not in stoplist])
    return sentence

def cleanStopword(documents, cleanTokens_once = False):

    stoplist = get_stopword_list(myConfig.stopword_file)
    texts = [[word for word in document.split() if word not in stoplist]
          for document in documents]
    if cleanTokens_once:
        all_tokens = sum(texts, [])
        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)

        texts = [[word for word in text if word not in tokens_once]
              for text in texts]

    logging.info('finish docs_preprocess')
    return texts


def preprocess(inputFile, output):

    fout = open(output, 'w')

     # for eachLine in open(inputJsonFile, 'r'):
    line_num = 0
    badline_couter = 0
    matcher = re.compile(r'本帖最后由.{1,48} 编辑\s*')
    for eachLine in codecs.open(inputFile, encoding='UTF-8'):
        line_num += 1

        if line_num % 1000 == 0:
            print line_num

        # print eachLine
        line = eachLine.strip().decode('utf-8')
        if line is u'':
            continue

        js = None
        try:
            js = json.loads(line)

            if 'contents'in js:
                content = js["contents"][0].strip().encode('utf-8')
                content = re.sub(matcher,'',content)
                if content is ':' or content is '：':
                    continue

                if content.startswith(':'):
                    content = content[1:].replace('\n','').strip()
                else:
                    content = content.replace('\n','').strip()

                content = seg(content)
                fout.write(content + '\n')

        except Exception, e:
            print 'bad line line_num %s:' %(line_num)
            print line
            print repr(line)
            badline_couter += 1
            print 'Bad line badline_couter: %s:' %(badline_couter)
            print 'Exception message: ' + e.message
            continue

    fout.close()
    print 'finish process %s lines' %(line_num)

def gen_dic_corpora():
    texts = preprocess.get_docs(myConfig.contents_seged_file)
    texts = preprocess.cleanStopword(texts)

    dictionary = corpora.Dictionary(texts)
    dictionary.save(myConfig.dict_file)

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(myConfig.corpora_file, corpus)

if __name__ == "__main__":
    preprocess('exdata\input_json.data', 'exdata\content-seged2.data')