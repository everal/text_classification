# -*- coding: utf8
#author:fangshu.chang
import numpy as np
import ConfigParser

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
#from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
import sklearn.naive_bayes
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#from sklearn.linear_model import SGDClassifier, LogisticRegression
import sklearn.linear_model
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model.sgd_fast import Hinge
from _mysql import result

try:
    import cPickle as pickle
except:
    import pickle

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

class Train():
    def __init__(self,configfilename):
        
        self.transet=[]
        self.tranlabel=[]
        self.testset=[]
        self.testlable=[]
        #设置模型保存路劲
        self.model_path=None
        
        #设置文本向量化参数
        self.ngram_range=None
        self.binary=None
        
        #设置卡方参数
        self.chi2=None
        self.chi2_num=None
        
        #设置tfidf参数
        self.tfidf=None
        self.smooth_idf=None
        self.sublinear_tf=None
        self.max_featrues=None
        self.idf=None
        
        #分离数据集为训练集和测试集
        self.test_size=None
        self.random_state=None
        
        #设置贝叶斯参数
        self.naive_bayes=None
        
        #设置线性模型参数
        self.linear_model=None
        self.loss=None
        self.penalty=None
        self.alpha=None
        self.n_iter=None
        self.class_weight=None
        self.C=None
        
        #设置网格搜索参数
        self.cv=None
        self.n_jobs=None
        self.verbose=None
        self.dictpara=None
        
        self.featuresfile=None
        self.config=ConfigParser.ConfigParser()
        self.cfgparser(configfilename)
        
        self.model=None
        
        #特征分析,特征按照权重排序
        self.features={}
        
    
    #解析配置文件
    def cfgparser(self,cfg_path):
        seclist=["default","preprocess","modelpara","resultpara"]
        optlist=["model_path","ngram_range","binary","chi2","chi2_num","tfidf","smooth_idf" \
                 ,"sublinear_tf","max_features","use_idf","naive_bayes","linear_model","sgd_loss" \
                 ,"sgd_penalty","sgd_alpha","sgd_n_iter","sgd_class_weight","lr_penalty","lr_c" \
                 ,"lr_class_weight","cv","n_jobs","verbose","dictpara"]
        self.config.read(cfg_path)
        if self.config.has_option("default", "features_path"):
            result=self.config.get("default", "features_path")
            if result is not None:
                self.featuresfile=result
            else:
                self.featuresfile="test_features"
        if self.config.has_option("default", "model_path"):
            result=self.config.get("default", "model_path")
            if result is not None:
                self.model_path=result
            else:
                self.model_path="test_model"
        if self.config.has_option("modelpara", "ngram_range"):
            result=self.config.get("modelpara", "ngram_range")
            if result is not None:
                self.ngram_range=tuple(eval(result))
            else:
                self.ngram_range=(1,1)
        if self.config.has_option("modelpara", "binary"):
            result=self.config.getboolean("modelpara", "binary")
            if result is not None:
                self.binary=result
            else:
                self.binary=False
        if self.config.has_option("modelpara", "chi2"):
            result=self.config.getboolean("modelpara", "chi2")
            if result is not None:
                self.chi2=result
            else:
                self.chi2=False
        if self.chi2:
            if self.config.has_option("modelpara", "chi2_num"):
                result=self.config.getint("modelpara", "chi2_num")
                if result is not None:
                    self.chi2_num=result
                else:
                    self.chi2_num=100
        if self.config.has_option("modelpara", "tfidf"):
            result=self.config.getboolean("modelpara", "tfidf")
            if result is not None:
                self.tfidf=result
            else:
                self.tfidf=False
        if self.tfidf:
            if self.config.has_option("modelpara", "smooth_idf"):
                result=self.config.getboolean("modelpara", "smooth_idf")
                if result is not None:
                    self.smooth_idf=result
                else:
                    self.smooth_idf=False
            if self.config.has_option("modelpara", "sublinear_tf"):
                result=self.config.getboolean("modelpara", "sublinear_tf")
                if result is not None:
                    self.sublinear_tf=result
                else:
                    self.sublinear_tf=False
            if self.config.has_option("modelpara", "max_features"):
                result=self.config.getint("modelpara", "max_features")
                if result is not None:
                    self.max_features=result
                else:
                    self.max_features=1000
            if self.config.has_option("modelpara", "use_idf"):
                result=self.config.getboolean("modelpara", "use_idf")
                if result is not None:
                    self.use_idf=result
                else:
                    self.use_idf=False
        if self.config.has_option("modelpara", "naive_bayes"):
            result=self.config.get("modelpara", "naive_bayes")
            if result is not None:
                self.naive_bayes=result
            else:
                self.naive_bayes=None
        if self.config.has_option("modelpara", "linear_model"):
            result=self.config.get("modelpara", "linear_model")
            if result is not None:
                self.linear_model=result
            else:
                self.linear_model=None
        if self.config.has_option("modelpara", "sgd_loss"):
            result=self.config.get("modelpara", "sgd_loss")
            if result is not None:
                self.loss=result
            else:
                self.loss="hinge"
        if self.config.has_option("modelpara", "sgd_penalty"):
            result=self.config.get("modelpara", "sgd_penalty")
            if result is not None:
                self.penalty=result
            else:
                self.penalty="l2"
        if self.config.has_option("modelpara", "sgd_alpha"):
            result=self.config.getfloat("modelpara", "sgd_alpha")
            if result is not None:
                self.alpha=result
            else:
                self.alpha=0.0001
        if self.config.has_option("modelpara", "sgd_n_iter"):
            result=self.config.getint("modelpara", "sgd_n_iter")
            if result is not None:
                self.n_iter=result
            else:
                self.n_iter=20
        if self.config.has_option("modelpara", "sgd_class_weight"):
            result=self.config.get("modelpara", "sgd_class_weight")
            if result is not None:
                self.class_weight=result
            else:
                self.class_weight=None
        if self.config.has_option("modelpara", "lr_penalty"):
            result=self.config.get("modelpara", "lr_penalty")
            if result is not None:
                self.penalty=result
            else:
                self.penalty="l2"
        if self.config.has_option("modelpara", "lr_c"):
            result=self.config.getfloat("modelpara", "lr_c")
            if result is not None:
                self.C=result
            else:
                self.C=1
        if self.config.has_option("modelpara", "lr_class_weight"):
            result=self.config.get("modelpara", "lr_class_weight")
            if result is not None:
                self.class_weight=result
            else:
                self.class_weight=None
        if self.config.has_option("modelpara", "cv"):
            result=self.config.getint("modelpara", "cv")
            if result is not None:
                self.cv=result
            else:
                self.cv=10
        if self.config.has_option("modelpara", "n_jobs"):
            result=self.config.getint("modelpara", "n_jobs")
            if result is not None:
                self.n_jobs=result
            else:
                self.n_jobs=2
        if self.config.has_option("modelpara", "verbose"):
            result=self.config.getint("modelpara", "verbose")
            if result is not None:
                self.verbose=result
            else:
                self.verbose=10
        if self.config.has_option("modelpara", "dictpara"):
            result=self.config.get("modelpara", "dictpara")
            if result is not None:
                self.dictpara=dict(eval(result))
            else:
                self.dictpara=None
        if self.config.has_option("modelpara", "test_size"):
            result=self.config.getfloat("modelpara", "test_size")
            if result is not None:
                self.test_size=result
            else:
                self.test_size=0.1
        if self.config.has_option("modelpara", "random_state"):
            result=self.config.getint("modelpara", "random_state")
            if result is not None:
                self.random_state=result
            else:
                self.random_state=10
    def train(self,X,y):
        self.transet,self.testset,self.tranlabel,self.testlable=train_test_split(X,y,test_size=self.test_size,random_state=self.random_state)
        methodlist=[]
        count_vec=CountVectorizer(ngram_range=self.ngram_range,binary=self.binary,decode_error="ignore")
        if self.tfidf:
            count_vec=TfidfVectorizer(ngram_range=self.ngram_range,binary=self.binary,smooth_idf=self.smooth_idf \
                                      ,sublinear_tf=self.sublinear_tf,use_idf=self.use_idf)
        methodlist.append(("count_vec",count_vec))
        chi=None
        if self.chi2:
            chi=SelectKBest(chi2,k=self.chi2_num)
            methodlist.append(("chi2",chi))
        clf=None
        if self.naive_bayes is not None:
            clf=getattr(sklearn.naive_bayes, self.naive_bayes)()
        else:
            try:
                clf=getattr(sklearn.linear_model, self.linear_model)(loss=self.loss,penalty=self.penalty,alpha=self.alpha \
                                                                     ,n_iter=self.n_iter,class_weight=self.class_weight)
            except Exception as e:
                clf=getattr(sklearn.linear_model, self.linear_model)(penalty=self.penalty,C=self.C,class_weight=self.class_weight)
        methodlist.append(("clf",clf))
        pipe=Pipeline(methodlist)
        grids=GridSearchCV(estimator=pipe,param_grid=self.dictpara,cv=self.cv,n_jobs=self.n_jobs,verbose=self.verbose)
        grids.fit(self.transet, self.tranlabel)
        result=grids.best_estimator_
        #print result.predict([X[2].replace("\t"," ").decode("utf8","ignore")])
        self.model=result
        try:
            fp = open(self.model_path,'wb')
        except IOError:
            print 'could not open file:',self.model_path
        pickle.dump(result, fp)
        fp.close()
        print classification_report(self.testlable,self.model.predict(self.testset))
       
       
    def featureanalysis(self):
        tmpindex=[]
        tmpdict=dict([(v,k) for (k,v) in self.model.get_params()["count_vec"].vocabulary_.iteritems()])
        tmpcoef=self.model.get_params()["clf"].coef_[0]
        if self.chi2:
            tmpindex=self.model.get_params()["chi2"].get_support(True)
            j=0
            for i in tmpindex:
                self.features[tmpdict[i]]=tmpcoef[j]
                j=j+1
        else:
            for i in range(len(tmpcoef)):
                self.features[tmpdict[i]]=tmpcoef[i]
        self.features=sorted(self.features.iteritems(), key=lambda x : x[1])
        
        featuresfile=open(self.featuresfile,"w")
        featuresfile.write("features count:"+str(len(self.features))+"\n")
        featuresfile.write(str(self.model.get_params()["clf"].intercept_)+"\n")
        for f in self.features:
            featuresfile.write(f[0]+" "+str(f[1])+"\n")
        featuresfile.close()
        
        
            
        
        
        
        
        