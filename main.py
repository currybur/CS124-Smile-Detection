import cv2
import numpy as np
from skimage.feature import local_binary_pattern as lbp
import os
from sklearn import svm
import time

dir = 'genki4k/files'
pics = os.listdir(dir)
file = open('genki4k/labels.txt', 'r')
label = file.readlines()
ts =3


def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier("./classifier/haarcascade_frontalface_default.xml")
    face = face_cascade.detectMultiScale(img, 1.2, 5)
    for (x, y, width, height) in face:
        result = (x, y, x+width, y+height)
        return result


def binFaces(image_name):
    face = detectFaces(image_name)
    image = cv2.imread(image_name)
    if face:
        the_face = image[face[1]:face[3], face[0]:face[2]]
        gray = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        return lbp(gray, 8, 1)
    else:
        return np.array([[None]])


def split_extract_image(src, rownum=10, colnum=10):
    l = []
    if binFaces(src).any() == None:
        return None
    else:
        img = binFaces(src)
        w, h = img.shape
        if rownum <= h and colnum <= w:
            num = 0
            row_height = h // rownum
            colwidth = w // colnum
            for r in range(rownum):
                for c in range(colnum):
                    box = (c * colwidth, r * row_height, (c + 1) * colwidth, (r + 1) * row_height)
                    (n, bins) = np.histogram(img[box[0]:box[2], box[1]:box[3]], bins=256, normed=True)
                    l += list(n)
                    num = num + 1
        li = l.copy()
        return li


def split_extract_noex(src, rownum=10, colnum=10):  # for predicting
        l = []
        img = binFaces_noex(src)
        w, h = img.shape
        if rownum <= h and colnum <= w:
            num = 0
            row_height = h // rownum
            colwidth = w // colnum
            for r in range(rownum):
                for c in range(colnum):
                    box = (c * colwidth, r * row_height, (c + 1) * colwidth, (r + 1) * row_height)
                    (n, bins) = np.histogram(img[box[0]:box[2], box[1]:box[3]], bins=256, normed=True)
                    l += list(n)
                    num = num + 1
        li = l.copy()
        return li


def binFaces_noex(image_name):
    faces = detectFaces(image_name)
    image = cv2.imread(image_name)
    if faces:
        the_face = image[faces[1]:faces[3], faces[0]:faces[2]]
        gray = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        return lbp(gray, 8, 1)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return lbp(gray, 8, 1)


def f1_score(result, a_label):
    tp = 0
    fn = 0
    for i in range(len(a_label)):
        if result[i] == 1 == a_label[i]:
            tp += 1
        elif result[i] == 0 == a_label[i]:
            fn += 1
    return 2*tp/(len(a_label) + tp - fn)



def get_data():
    sta = time.time()
    train_data_list = []
    test_data_list = []
    count = 0
    train_data = open('./data/train_data.txt','w')
    test_data = open('./data/test_data.txt','w')
    labels = open('./data/labels.txt','w')
    print('\n\nDrawing features, please wait for a moment patiently...')
    time.sleep(ts)
    for i in range(4000):
        feature1 = split_extract_image(dir + os.sep + pics[i])
        feature2 = split_extract_noex(dir + os.sep + pics[i])
        train_data_list.append(feature1)
        test_data_list.append(feature2)
        if feature1 == None:
            train_data.write('None')
        else:
            train_data.write(str(feature1))
        train_data.write('\n')
        test_data.write(str(feature2))
        test_data.write('\n')
        labels.write(str(label[i][0]))
        labels.write('\n')
        count += 1
        if count%4 == 0:
            print('%.1f%% loaded'%(count*100/4000))
    train_data.close()
    test_data.close()
    labels.close()
    print('Finished. Elapsed time: %.1f'%(time.time() - sta))


def train_test():
    print('Loading data......')
    sta = time.time()
    train_data_list = []
    test_data_list = []
    lab = []

    for m in open('./data/train_data.txt','r').readlines():
        train_data_list.append(eval(m))

    for n in open('./data/test_data.txt','r').readlines():
        test_data_list.append(eval(n))

    for num in open('labels.txt','r').readlines():
        lab.append(int(num))

    print('\nData is ready. Elapsed time: %.1f'%(time.time() - sta))
    time.sleep(ts)
    all = [0]*10
    f1_scores = [0]*10

    for i in range(10):
        print('\nStart to train detector No.%d \n'%(i+1))
        time.sleep(ts)
        train_data = []
        train_label = []
        test_data = test_data_list.copy()
        test_label = lab.copy()
        count = 0

        for k in range(0+i, 4000, 10):
            if train_data_list[k] == None:
                pass
            else:
                train_data.append(train_data_list[k])
                train_label.append(lab[k])
            del(test_data[k-count])
            del(test_label[k-count])
            count += 1
            if count%4==0:
                print('Progress:%.1f%%'%(100*count/400))

        clf = svm.LinearSVC()
        clf.fit(train_data,train_label)
        #joblib.dump(clf, 'smile_detector_plus'+str(i+1)+'.pkl')
        print('\nDetector No.%d finished. Begin to test...'%(i+1))
        time.sleep(ts)
        result = clf.predict(test_data)
        all[i] = np.mean(np.array(test_label)==result)
        f1_scores[i] = f1_score(result,test_label)
        print('\nResult: fold %d :accuracy: %.2f; f1score: %.2f'%(i +1,all[i],f1_scores[i]))

    aa = np.mean(all)
    af = np.mean(f1_scores)
    print('\n\nThe average accuracy is %.2f; the average f1score is %.2f'%(aa,af))

get_data()
train_test()


