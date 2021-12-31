"""
@Time : 2021/5/31 10:33
@Author : wmingzhu
@Annotation : 
"""
import re,json
import numpy as np
from PIL import Image
import io
import logging
#将图片转为numpy数组
"""
PIL读出来的图片size应该是(width,height)，但是转成numpy矩阵后，变成了(height, width, channels)。所以resize的时候应该注意调换位置。numpy与tensorflow一致
"""
def universal_image_process(imagepath:str,shape:list)->np.array:
    print("tag:universal_image_process正在处理图片,将图片转为numpy数组")
    image = Image.open(imagepath)
    try:
        len_shape = len(shape)
        image_array = np.array(image)#将PIL.JpegImagePlugin.JpegImageFile对象转为numpy数组
        # 如果输入张量是一维的，那么需要把图片拉平为一维的
        if len_shape == 1:
            #如果图片是3通道的RGB图片
            if len(image_array.shape) == 3:
                print("3通道图片处理成1维张量")
                #把图片resize到shape要求的尺寸大小
                quotient = int(shape[0] / 3)#这个quotient*3比shape[0]少的部分后面补齐
                image2 = image.resize((quotient,1))
                image2 = np.array(image2)#image2.shape->(1,quotient,3)
                product = 1
                for i in range(3):
                    product *= image2.shape[i]
                image2.shape = (product,)
                image2 = list(image2)
                #把不够的部分补上
                for i in range(shape[0]-product):
                    image2.append(127)#取255的一半
                image2 = np.array(image2,dtype="float32")
                image2 /= 255.0
                image2.shape = (1,) + image2.shape
                return image2
            elif len(image_array.shape) == 2:
                print("把灰度图片处理成一维张量")
                image3 = image.resize((shape[0],1))
                image3 = np.array(image3,dtype="float32")
                image3 /= 255.0
                return image3
        elif len_shape == 2:
            if len(image_array.shape) == 3:
                print("把3通道图片处理成二维张量")
                image4 = image.resize((int(shape[0]*shape[1]/3),1))
                image4 = np.array(image4)
                image4.shape = (int(shape[0]*shape[1]/3)*3,)
                image4 = list(image4)
                for i in range(shape[0]*shape[1]-int(shape[0]*shape[1]/3)*3):
                    image4.append(127)
                image4 = np.array(image4,dtype="float32")
                image4.shape = (1,)+tuple(shape)
                image4 /= 255.0
                return image4
            elif len(image_array.shape) == 2:
                print("把灰度图片处理成二维张量")
                image5 = image.resize((shape[1],shape[0]))
                image5 = np.array(image5,dtype="float32")
                print("图片形状为:",image5.shape)
                image5 /= 255.0
                image5.shape = (1,)+tuple(shape)
                return image5
        elif len_shape == 3:
            if len(image_array.shape) == 3:
                print("把3通道的转为3维张量")
                #如果上传的图片是三通道，而shape是类似[32,32,1]这种形式的话，那么转化肯定会失败，只能提前把图片转为灰度的
                if shape[2] == 1:
                    print("转为灰度前的形状为:",np.array(image,dtype="float32").shape)
                    image6 = image.convert("L")
                    print("如果这个shape的最后一维为1，那么需要把三通的图片先转为灰度图片")
                    # image6.save("singleChannel.png")
                    image_test = np.array(image6,dtype="float32")
                    print("转为灰度后的形状为:",image_test.shape)
                    image6 = image6.resize((shape[1],shape[0]))
                    image6 = np.array(image6,dtype="float32")
                    image6 /= 255.0
                    image6.shape = (1,)+tuple(shape)
                    return image6
                elif shape[2] == 3:
                    print("如果shape的第三维度值本身就是3，那么直接将三通到的图片resize成shape要求的形状即可")
                    image6 = image.resize((shape[1], shape[0]))
                    image6 = np.array(image6, dtype="float32")
                    image6 /= 255.0
                    image6.shape = (1,) + tuple(shape)
                    return image6
            if len(image_array.shape) == 2:
                print("把单通道的转为3维张量")
                if shape[2] == 1:
                    print("三维张量shape的最后一维为1")
                    print("灰度图片resize前的形状为", np.array(image, dtype="float32").shape)
                    image7 = image.resize((shape[1],shape[0]))
                    image7 = np.array(image7, dtype="float32")
                    image7 /= 255.0
                    image7.shape = (1,) + tuple(shape)
                elif shape[2] == 3:
                    image7 = image.convert("RGB")#如果shape[2]==3，因为image是单通道的，所以还要将其转为三通道的
                    image7 = image7.resize((shape[1], shape[0]))
                    image7 = np.array(image7, dtype="float32")
                    image7 /= 255.0
                    image7.shape = (1,) + tuple(shape)
                return image7
    except Exception as e:
        print("错误:",e)


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
#同上
def savemodel_input_shape(model_json:str,logger=None)-> list:
    model_structure = json.loads(model_json)
    # 通过反向预查来匹配模型输入
    pattern = re.compile('(?<=\'batch_input_shape\': ).*?\]')
    result = pattern.search(str(model_structure["config"])).group()
    pattern2 = re.compile("\d+?(?=,|\])")
    shape = pattern2.findall(result)
    shape = [int(element) for element in shape]
    return shape