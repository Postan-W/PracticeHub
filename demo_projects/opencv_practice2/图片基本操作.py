import cv2
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

split_merge()