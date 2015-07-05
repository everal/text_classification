#coding=utf8
import re

def seg(WeiboContent):
    content=""
    m=re.search("//@[\w\W]+?:",WeiboContent)
    if m:
        index=WeiboContent.find(m.group(0))
        content=WeiboContent[:index]
        return content
    else:
        return WeiboContent

def clean(WeiboContent):
    content=WeiboContent
    l_clean=[]
    #l_clean.append("回复@[\w\W]+?:")    #reply someone in the front of the content
    l_clean.append("//@[\w\W]+?:")       #follow someone behind the content
    l_clean.append("@[\w\W]+? ")         #remind someone in the content
    l_clean.append("http://[0-9A-Za-z\./]*")
    l_clean.append("来自[\w\W]+[ ）)。/]")
    l_clean.append("via[\w\W]+[ ）)。/]")
    l_clean.append("分享自[\w\W]+[ ）)。/]")
    l_clean.append("微信：[0-9A-Za-z]*")
    l_clean.append("[vV]信：[0-9A-Za-z]*")
    l_clean.append("[0-9]{11}")
    for i in l_clean:
        (content,num)=re.subn(i,"",content)
        #print content
    return content

if __name__ == '__main__':
    content="我@大灰狼 独自走在郊外的小路上http://t.cn/adsgf13jgv我把糕点带给外婆尝一尝。//@小红帽:鼓励下[鼓掌]"
    content=seg(content)
    print content
    content=clean(content)
    print content
