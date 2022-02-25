import cv2
from PIL import Image
video_path = r"C:\Users\15216\Videos\mda-mf2cxm4j09ms8ywu.mp4"
video = cv2.VideoCapture(video_path)
ret,first_frame = video.read()
image = Image.fromarray(first_frame)
print(ret)
image.show()
