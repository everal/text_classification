# -*- coding: utf8
#author:fangshu.chang
import ConfigParser
import sys

import Predict
import Mypreprocess
import Train

class Main():
    def __init__(self,configfilename):
        self.method=None
        self.data=None
        
        self.train=Train.Train(configfilename)
        self.predict=Predict.Predic(configfilename)
        self.preprocess=Mypreprocess.Mypreprocess(configfilename)
        
        self.config=ConfigParser.ConfigParser()
        self.cfgparser(configfilename)
    
    def cfgparser(self,cfg_path):
        seclist=["preprocess"]
        optlist=["method"]
        self.config.read(cfg_path)
        if self.config.has_option("preprocess", "method"):
            result=self.config.get("preprocess", "method")
            if result is not None:
                self.method=result
            else:
                self.method="simpleprocess"
        if self.config.has_option("preprocess", "data"):
            result=self.config.get("preprocess", "data")
            if result is not None:
                self.data=result
            else:
                self.data=None
        
    def main(self,text):
        getattr(self.preprocess, self.method)()
        self.train.train(self.preprocess.dataset,self.preprocess.label)
        self.train.featureanalysis()
        self.predict.predict(self.preprocess.datatopredict)
        print self.predict.predictvalue


if __name__=='__main__':
    if len(sys.argv) != 2:
        print "useage: python Main.py example.cfg"
    else:
        test=Main(sys.argv[1])
        test.main("买 一 台 三菱 空调 ， 带 着 安装 师傅 上 上 电梯 ， 结果 被 一个 老头 给 鄙\
    视 了 ， 他 看 了 师傅 穿 的 工作服 logo 说道 ： 中国 人 就 是 贱 ， 一边 骂 日本 人 ，\
     一边 又 买 日货 ！ 我 转头 看 着 他 ， 他 接着 说 ： 难道 不 是 吗 ， 现在 连 欧 美 货 \
     都 比 不 上 小 日本 的 ！ 大家 都 在 骂 ， 又 都 没 办法 不 买 ！ 最后 这 句 说 到 重点 了 ！")
