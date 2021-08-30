# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from skimage import exposure
from skimage import feature
from imutils import paths
import pywt
import pywt.data
import argparse
import imutils
import cv2
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
start_time = time.time()
print(start_time)
# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="training\\")
ap.add_argument("-t", "--test", required=True, help="test\\")
args = vars(ap.parse_args())
 
# initialize the data matrix and labels
print("[INFO] extracting features...")
data = []
des = []
labels = []

for imagePath in paths.list_images(args["training"]):
    # extract the make of the car
    make = imagePath.split("\\")[1]
    print(imagePath)
 
    # load the image, convert it to grayscale, and detect edges
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # create SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create(20)

    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(gray,None)
    des = [descriptors[1], descriptors[5],
                  descriptors[10],descriptors[13],descriptors[16]]
    ga1= np.array(des)
    ga2 = ga1.flatten()
    data.append(ga2)
    labels.append(make)

##print(data)
# "train" the nearest neighbors classifier
print("[INFO] training classifier...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.70,test_size = 0.30)
print(len(X_train), len(X_test), len(y_train), len(y_test))


##model = KNeighborsClassifier(n_neighbors=1)
#kfold = KFold(n_splits=10)
model = SVC(kernel='linear',gamma = 0.142, probability=True) # poly, sigmoid, rbf
##model = KNeighborsClassifier(n_neighbors=1)
#results = cross_val_score(model,X_train,y_train, cv = kfold)
model.fit(X_train, y_train)
print("[INFO] evaluating...")

y_pred = model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)
#print(results)
#print('Accuracy = {}%, Standard Deviation = {}%'.format(round(results.max(), 4), round(results.std(), 2)))
end_time = time.time()
print("Total execution time: {} seconds".format(end_time - start_time))



des1 =[]


for (i, imagePath) in enumerate(paths.list_images(args["test"])):
    # load the test image, convert it to grayscale, and resize it to
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #logo = cv2.resize(gray, (200, 100))

    # create SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create(20)

    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(gray,None)
    des1 = [descriptors[1], descriptors[5],
                  descriptors[10],descriptors[13],descriptors[16]]
    ga1 = np.array(des1)
    ga2 = ga1.flatten()

    pred = model.predict(ga2.reshape(1, -1))[0]
 
    # draw the prediction on the test image and display it
    cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255 ), 3)
    cv2.imshow("Test Image #{}".format(i + 1), image)
    cv2.waitKey(0)


