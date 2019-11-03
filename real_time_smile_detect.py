import cv2
from skimage.feature import local_binary_pattern as lbp
import numpy as np
from sklearn.externals import joblib
import time

clf = joblib.load('smile_detector5.pkl')

def detect_faces(img):
    faces = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x+width, y+height))
    return faces,result


def get_all_faces(img):
    a,faces = detect_faces(img)
    faces_list = []
    for face in faces:
        the_face = img[face[1]:face[3], face[0]:face[2]]
        faces_list.append(lbp(the_face,8,1))
    return a,faces_list


def split_face(image, rownum=8, colnum=8):
    l = []
    img = image
    w, h = img.shape
    num = 0
    row_height = h // rownum
    colwidth = w // colnum
    for r in range(rownum):
        for c in range(colnum):
            box = (c * colwidth, r * row_height, (c + 1) * colwidth, (r + 1) * row_height)
            (n, bins) = np.histogram(img[box[0]:box[2],box[1]:box[3]], bins=256, normed=True)
            l += list(n)
            num = num + 1
    return l


def judge_smile(face):
    x = split_face(face)
    ans = clf.predict([x])
    return ans

cap = cv2.VideoCapture(0)

classfier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

color = (239, 255, 210)
font=cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
        sta = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects,facelist = get_all_faces(grey)
        if len(faceRects) > 0:
            for i in range(len(faceRects)):
                x, y, w, h = faceRects[i]
                cv2.rectangle(frame, (x , y ), (x + w - 5 , y + h - 5), color, 1)
                if len(facelist) > 0:
                        if judge_smile(facelist[i])[0] == 1:
                            cv2.putText(frame,'smiling',(x,y-10),font,1,(239,255,210),1)
                        else: #judge_smile(face)[0] == 0:
                            cv2.putText(frame,'not smiling',(x,y-10),font,1,(239,255,210),1)

        cv2.imshow('Monster-reflecting Mirror', frame)
        c = cv2.waitKey(1)
        print('FPS:%.d'%(1/(time.time()-sta)))
        if c ==27:
            break
cap.release()
cv2.destroyAllWindows()
