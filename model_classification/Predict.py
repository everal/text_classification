# -*- coding: utf8
#author:fangshu.chang

import ConfigParser
try:
    import cPickle as pickle
except:
    import pickle
class Predic():
    def __init__(self,configfilename):
        self.predictvalue=None
        self.predictprobavalue=None
        self.model_path=None
        
        self.config=ConfigParser.ConfigParser()
        self.cfgparser(configfilename)
    
    def cfgparser(self,cfg_path):
        seclist=["default"]
        optlist=["model_path"]
        self.config.read(cfg_path)
        result=self.config.get("default", "model_path")
        if result is not None:
            self.model_path=result
        else:
            self.model_path="test_model"
    def predict(self,text):
        
        fp = open(self.model_path,'r')
        clf=pickle.load(fp)
        fp.close()
        self.predictvalue=clf.predict(text)
        
