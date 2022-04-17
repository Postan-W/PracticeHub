"""
Date: 2022/04/17
"""
import urllib.request as ur
from configparser import ConfigParser
import urllib.parse as uparse

config = ConfigParser()
config.read("./urls.ini")
url = config.get("url","baidu")

#获取网页信息
def get_full(url):
    response = ur.urlopen(url)
    print(type(response))#HTTPResponse
    print(response.getcode())#状态码
    print(response.getheaders())#获取头
    print(response.geturl())#获取url
    # content为bytes类型
    content = response.readline()#读取一行
    print(content)
    content = response.read(20)#读取20个字节
    print(content)
    content = response.readlines()#返回包含所有行的列表
    # print(content)
    content = response.read().decode('utf-8')#默认获取整个网页

# get_full(url)

#下载
def download_from_url(url,filename):
    #第一个参数是url,第二个参数是文件保存位置
    ur.urlretrieve(url,filename)

# download_from_url(url,"./files/texts/baidu.html")

header1 = {'User-Agent':
'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'}
#定制请求对象
def custom_request(url,header):
    request = ur.Request(url=url,headers=header)
    response = ur.urlopen(request)
    content = response.read().decode('utf-8')
    print(content)

#将字符串转为Unicode编码形式
custom_request(config.get("url","jaychou")+uparse.quote("周杰伦"),header1)

