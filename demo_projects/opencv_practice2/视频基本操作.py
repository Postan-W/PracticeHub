import cv2
import matplotlib.pyplot as plt
#示例函数
def save_video(rtsp_url):
    """
    视频的编码格式参考如下：
    cv2.VideoWriter_fourcc('I','4','2','0'):YUV编码，4:2:0色度子采样。这种编码广泛兼容，但会产生大文件。文件扩展名应为.avi。
    cv2.VideoWriter_fourcc('P','I','M','1'):MPEG-1编码。文件扩展名应为.avi。
    cv2.VideoWriter_fourcc('X','V','I','D'):MPEG-4编码。如果要限制结果视频的大小，这是一个很好的选择。文件扩展名应为.avi。
    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'):较旧的MPEG-4编码。如果要限制结果视频的大小，这是一个很好的选择。文件扩展名应为.m4v。
    cv2.VideoWriter_fourcc('X','2','6','4'):较新的MPEG-4编码。如果你想限制结果视频的大小，这可能是最好的选择。文件扩展名应为.mp4。
    cv2.VideoWriter_fourcc('T','H','E','O'):这个选项是Ogg Vorbis。文件扩展名应为.ogv。
    cv2.VideoWriter_fourcc('F','L','V','1'):此选项为Flash视频。文件扩展名应为.flv。
    :param rtsp_url: 视频流地址
    :return:
    """
    cap = cv2.VideoCapture(rtsp_url)
    # 获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 限制视频存储的画面大小
    size = (750, 420)
    # 设置视频的编码格式
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 定义视频保存的输出属性
    out = cv2.VideoWriter('rt_out.avi', fourcc, fps, size)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)
    finally:
        try:
            cv2.destroyWindow("frame")
            cap.release()
        except Exception as e:
            print(e)

video = cv2.VideoCapture("./videos/hotelCalifornia.mp4")
fps = video.get(5)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_writer = cv2.VideoWriter("./videos/hotelCalifornia_gray.mp4",cv2.VideoWriter_fourcc('X','2','6','4'),fps,size)
while video.isOpened():
    ret,frame = video.read()
    if not ret:
        continue
    #这里把图片转为灰度图
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("hotel california",frame)
    #请注意&是位运算，优先级比==高；而and是逻辑运算，优先级比==低。
    """
    对下面的if条件句作下解释:
    使用 cv2.waitKey阻塞一定毫秒数，等待键入某个键来结束阻塞，这里设置的是10ms，超过设置的时间阻塞也会结束(要不然因为read是非常快的，imshow也是很快，会出现不理想画面)，
    cv2.waitKey的返回值就是键入的那个键值(可以验证print(cv2.waitKey(10))，如果没有键入值，那么就会等阻塞时间到了返回一个-1)，一般来说返回值不止8位，比如32位，而键盘上的键都是用8位来表示的，所以前面24位都是0，后面8位才是有用的，
    所以这里使用&与0xFF做位操作，0xFF这个16进制数即255,即二进制的11111111，所以做完&的结果就是返回值的最后8位，0xFF也可以被看作是掩码；ord内建函数的
    作用是返回一个字符的Unicode码，当然对于字符来说等同于ascii码，即一个8位的二进制码；所以==左边是你键入的键值，右边是设定的一个字符，如果一样，这里是执行了break
    """
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    video_writer.write(frame)
video.release()
cv2.destroyAllWindows()
