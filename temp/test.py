import requests
import os
from zipfile import ZipFile
def download():
    url = "http://rtdev.bonc.tech/datasience/xquery" + '/dsModel/downloadModelForModelService'
    params = {
        "modelId": "d96ed878-3864-40cc-a831-fa3925e3b09c",
        "version": "V1",
        "guid": "datascience"
    }
    try:
        res = requests.get(url, params=params)
        with open("model.zip", 'wb') as f:
            f.write(res.content)
    except requests.exceptions.Timeout as e:
        print("模型下载错误{}".format(e))

# download()
def uncompress():
    with ZipFile("testmodel.zip", 'r') as f:
        zips = list(filter(lambda x: x.startswith("model/model"), f.namelist()))
        f.extractall("./model",members=zips)
        print("压缩文件内部情况是:", zips)

uncompress()
#read pyspark csv
# from pyspark import SparkContext,SparkConf
# from pyspark.sql import SparkSession,Row,SQLContext
# conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
# # sc=SparkContext.getOrCreate(conf)
# spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
# sc = spark.sparkContext
#
# def read_file(path:str):
#    data_frame =  spark.read.format("csv").option('header','true').load("file:///"+path)
#    data_frame.show()
#    pandas_frame = data_frame.toPandas()
#    print(pandas_frame)
#    return pandas_frame
# read_file()