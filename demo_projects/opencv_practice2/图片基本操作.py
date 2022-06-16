import cv2
import matplotlib.pyplot as plt
#裁剪图片,计算机是左上角为(0,0)点，y和x(或者说h和w)分别是向下和向右增加
def cut():
    image = cv2.imread("./images/8.jpg")
    image = image[300:1000, 500:1000]  # 先后对h,w进行裁剪
    cv2.imshow("无题", image)
    cv2.waitKey(0)

#抽离通道与合并通道。cv2图片对象的通道顺序是BGR
def split_merge():
    image = cv2.imread("./images/8.jpg")
    b,g,r = cv2.split(image)
    print(b.shape,g.shape,r.shape)
    merge_bgr = cv2.merge((b,g,r))
    print(merge_bgr.shape)
    #演示只保留R通道
    merge_bgr[:,:,0] = 0#b通道的值均设为0
    merge_bgr[:,:,1] = 0#g通道的值均设为0
    cv2.imshow("only r",merge_bgr)
    cv2.waitKey(0)

#边界填充
def fill_border(img_path):
    top,bottom,left,right = (200,500,200,200)
    img = cv2.imread(img_path)
    #复制最边缘像素
    replicate = cv2.copyMakeBorder(img,top,bottom,left,right,borderType=cv2.BORDER_REPLICATE)
    #以最边缘为轴作镜面对称
    reflect = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)
    #以最边缘靠里一点为轴作镜面对称，例如edcb|abcde|dcba
    reflect101 = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
    #以本例为例，把从上到下200像素的画面补给下面，从下到上200像素的画面补给上面，从左到右的补给右边，从右到左的补给左边
    wrap = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_WRAP)
    #常值填充，比如value1=0就是填黑色
    constant = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,value=0)
    #默认是显示RGB图像，但cv2图像对象通道顺序是BGR，显示出来不好看，所以这里干脆显示灰度图
    plt.subplot(231),plt.imshow(img,"gray"),plt.title("origin")
    plt.subplot(232), plt.imshow(replicate, "gray"), plt.title("replicate")
    plt.subplot(233), plt.imshow(reflect, "gray"), plt.title("reflect")
    plt.subplot(234), plt.imshow(reflect101, "gray"), plt.title("reflect101")
    plt.subplot(235), plt.imshow(wrap, "gray"), plt.title("wrap")
    plt.subplot(236), plt.imshow(constant, "gray"), plt.title("constant")
    plt.show()

#图像的数值计算。图像的resize。图像的融合
def calc(img_path1,img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    print(type(img1),img1.dtype)#<class 'numpy.ndarray'>。所以可以利用numpy.ndarray的特性作运算。dtype为uint8正好符合图像数值0-255。大于255则以256为模取模
    print(img1[:5,:,0])
    img1_plus10 = img1 + 10
    print(img1_plus10[:5,:,0])
    #注意numpy的shape是HW顺序，cv2.resize的顺序是WH
    img3 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))
    print(img1.shape,img3.shape)
    #还可以按倍率resize，此时WH指定为(0,0)
    img4 = cv2.resize(img2,(0,0),fx=3,fy=1)
    plt.imshow(img4),plt.show()

    #两个相同大小的图片按照ax1+bx2+b进行融合
    combine = cv2.addWeighted(img1,0.4,img3,0.6,0)
    plt.imshow(combine),plt.show()


calc("./images/google.jpeg","./images/cat.jpeg")
