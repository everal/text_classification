# -*- coding: utf8
#author:fangshu.chang
import ConfigParser
#import sys

import nlpir

#reload(sys)
#sys.setdefaultencoding('utf8')
class Preprocess():
    def __init__(self,configfilename):
        
        #这些变量中数据必须是分好词后的数据
        #训练模型数据
        self.dataset_path=None
        self.dataset=[]
        self.label=[]
        
        #要预测的数据
        self.datatopredict=[]
        
        self.config=ConfigParser.ConfigParser()
        self.cfgparser(configfilename)
    
    def cfgparser(self,cfg_path):
        seclist=["default"]
        optlist=["dataset_path"]
        self.config.read(cfg_path)
        if self.config.has_option("default", "dataset_path"):
            result=self.config.get("default", "dataset_path")
            if result is not None:
                self.dataset_path=result
            else:
                self.dataset_path="test_dataset"
            
    #继承此类需实现该方法,该方法中必须给self.dataset和self.label赋值
    def readdataset(self,data):
        pass
    
    #继承此类需实现该方法，该方法中必须给self.datatopredict赋值
    def readdatatopredict(self,data):
        pass
    
    #数据文件的格式是text \t label
    #使用中科院分词工具进行分词
    #结果保存在self.dataset和self.label列表中
    def simpleprocess(self):
        data=open(self.dataset_path,"r")
        for d in data.readlines():
            item=d.strip().split("\t")
            self.dataset.append(nlpir.segWithTag(item[0]).replace("\t", " "))
            self.label.append(item[1])
        data.close()
        
    #数据文件的格式是label#$|_$_|$#品牌#$|_$_|$#text
    #使用中科院分词工具进行分词
    #结果保存在self.dataset和self.label列表中
    def simpleprocessone(self):
        data=open(self.dataset_path,"r")
        for d in data.readlines():
            item=d.strip("\n").split("#$|_$_|$#")
            if len(item)==3:
                self.dataset.append(nlpir.segWithTag(item[2]).replace("\t", " "))
                self.label.append(item[0])
        data.close()
            
        