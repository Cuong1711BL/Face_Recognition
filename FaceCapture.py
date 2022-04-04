import cv2
from facenet_pytorch import MTCNN
from datetime import datetime
import os


IMG_PATH = './data/test_images/'
count = 200
usr_name = input("Input ur name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)


mtcnn = MTCNN(margin = 10, keep_all=False, select_largest = True, post_process=True)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while cap.isOpened() and count:
    isSuccess, frame = cap.read()
    if mtcnn(frame) is not None:
        path = str(USR_PATH+'/{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-")+"-"+str(count)))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        face_img = mtcnn(frame, save_path = path)
        count-=1
    cv2.imshow('Face Capturing', frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()