# -*- coding: utf8
#author:fangshu.chang

"""
本预测函数只适用于逻辑回归，且词汇特征的取值为词频
"""
from math import exp

class LRPredict():
    def __init__(self,featuresfile):
        #特征是Key，权重是Value
        self.featuresdict={}
        #常量
        self.w0=None
        
        self.floatvalue=None
        self.truevalue=None
        self.intvalue=None
        
        self.setfeatures(featuresfile)
        
    #输入：特征文件，格式：
    #features count:3000
    #[-0.53936547]
    #视频 -3.18484058882
    #行业 -3.0937104435
    #输出：设置self.featuresdict和self.w0
    def setfeatures(self,featuresfile):
        ff=open(featuresfile,"r")
        ff.readline()
        self.w0=float(ff.readline().strip("\n[]"))
        for f in ff.readlines():
            tmpf=f.strip().split(" ")
            self.featuresdict[tmpf[0]]=float(tmpf[1])
        ff.close()
    
    #输入：文本，格式：词汇 词汇 词汇
    #输出：预测值，float类型
    def predict(self,text):
        result=self.w0
        textlist=text.strip().split()
        for t in textlist:
            if self.featuresdict.has_key(t):
                result+=self.featuresdict[t]
        print result
        self.floatvalue=1/(1+exp(-result))
        if self.floatvalue>0.5:
            self.intvalue=1
            self.truevalue=True
        else:
            self.intvalue=0
            self.truevalue=False
        return self.floatvalue

if __name__=='__main__':
    test=LRPredict("test_features")
    print test.predict("? 为什么 一 线 和 三 线 的 移动 公司 营业厅 被 出租车 包围 ？")
    print test.intvalue
    print test.truevalue
    
        
    