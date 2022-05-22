from PIL import Image
import numpy as np
import pandas as pd
import os
from keras.datasets import mnist
path = "C:\\Users\\15216\\Desktop\\datasets\\classification\\mnist3_big\\"
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
images = []
for i in range(len(test_images)):
    file_name = str(i) + ".jpg"
    images.append(file_name)
    entire_path = path + "test_image\\" + file_name
    image = Image.fromarray(test_images[i])
    image.save(entire_path)

dict1 = {"image":images,"label":test_labels}
dataframe = pd.DataFrame(dict1)
dataframe.to_csv(path+"test_label\\"+"test_tag.csv")

