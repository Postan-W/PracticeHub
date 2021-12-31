import re,json

import numpy
import numpy as np
from PIL import Image

def h5_input_shape(model_json:str,logger=None)-> list:
    model_structure = json.loads(model_json)
    # 通过反向预查来匹配模型输入
    pattern = re.compile('(?<=\'batch_input_shape\': ).*?\]')
    #上面的正则的意思依次是：反向预查，匹配除“\n”和"\r"之外的任何单个字符，任意多次，非贪婪模式(即把每次匹配的机会先让给后面),以]结束
    result = pattern.search(str(model_structure["config"])).group()
    pattern2 = re.compile("\d+?(?=,|\])")
    shape = pattern2.findall(result)
    shape = [int(element) for element in shape]
    return shape

#将预测图片转为模型接收的形状，模型接收的形状用参数shape指定：[H,W,C]
def image_resize(image_path:str,shape:list=None)->numpy.array:
    image = Image.open(image_path)
    print("图片的格式:{}     尺寸:{}     模式:{}".format(image.format,{"width":image.size[0],"height":image.size[1]},image.mode))
    # 如果模型处理的是单通道图，那么将其转为灰度图
    if shape[2] == 1:
        print("=====转为灰度图=====")
        image = image.convert("L")
    #Image读取的图片以及相关操作比如下面的resize，都是W,H的顺序
    image = image.resize((shape[1],shape[0]))
    image_numpy = np.array(image)
    print("转为numpy后的形状为:{}".format(image_numpy.shape))
    # image.show()
    print(image_numpy.reshape((shape[2],shape[0],shape[1])))
    return image_numpy#如果是灰度图则返回的numpy中不包含通道维度

image_resize("test_images/animal.jpg",shape=[400,500,3])