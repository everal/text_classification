# -*- coding: utf-8 -*-
import sys
import re
import clean
reload(sys)
sys.setdefaultencoding('utf-8')

class Classifier():
    def __init__(self,ruleFile):
        self.ruleFile = ruleFile
        self.d_Attribute, self.threshold = self.setRule(self.ruleFile)

    def setRule(self,ruleFile):
        d_Attribute = {}
        threshold = 0
        f = open(ruleFile, 'r')
        texts = f.readlines()
        for line in texts:
            line = line.strip()
            if line.startswith("bfd"):
                line = line.split(' ')
                k = line[1]
                v = line[2].split('，')
                d_Attribute[k] = v
            else:
                line = line.strip()
                line = line.split('=')
                threshold = float(line[1].strip('\n'))
        return d_Attribute, threshold

    def lookup(self,l_words,text):
        flag = 0
        for word in l_words:
            m = re.search(word, text)
            if m:
                flag = 1
                return flag
        return flag

    def isValid(self,wbcontent):
        #clean the content
        wbcontent = clean.seg(wbcontent)
        wbcontent = clean.clean(wbcontent)
        #calcuate the weight
        key_total = 0.0
        for key in self.d_Attribute:
            flag = self.lookup(self.d_Attribute[key],wbcontent)
            if flag:
                for words in self.d_Attribute[key]:
                    patt = re.compile(words)
                    sk_num = len(patt.findall(wbcontent))*float(key)
                    key_total = key_total+sk_num
        if key_total > self.threshold or key_total == self.threshold:
            key_total = 1.0
        else:
            key_total = 0.0
        return  key_total

if __name__ == "__main__":
    r = Classifier('rule.txt')
    print r.isValid("顺丰圆通")
    
