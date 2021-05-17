import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image 
import PIL.ImageOps


X  = np.load("image.npz")['arr_0']
y = pd.read_csv("Pro123 - Data.csv")["labels"]
print(pd.Series(y).value_counts())
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
n = len(labels)
print(n)

x_train , x_test , y_train , y_test = train_test_split(X , y , random_state = 9 , train_size = 7500 , test_size = 2500)
x_train_scl = x_train/255
x_test_scl = x_test/255

clf = LogisticRegression(solver="saga" , multi_class="multinomial" ).fit(x_train_scl , y_train)
y_pred = clf.predict(x_test_scl)
accuracy = accuracy_score(y_test , y_pred)
print(accuracy)

capture = cv2.VideoCapture(0)
while(True):
    try:
        ret , frame = capture.read()
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        height , width = gray.shape
        upperleft = (int(width/2 - 56) , int(height/2 - 56))
        bottomRight = (int(width/2 + 56) , int(height/2 + 56))
        cv2.rectangle(gray , upperleft , bottomRight , (0 , 255 , 0) , 2)
        roi = gray[upperleft[1]:bottomRight[1] , upperleft[0]:bottomRight[0]]
        impil = Image.fromarray(roi)
        imagebw = impil.convert('L')
        imagebwresize = imagebw.resize((28 , 28) , Image.ANTIALIAS)
        inverted = PIL.ImageOps.invert(imagebwresize)
        filter = 20
        minpix = np.percentile(inverted , filter)
        inv_scl = np.clip(inverted-minpix , 0 , 255)
        maxpix = np.max(inverted)
        inv_scl = np.asarray(inv_scl/maxpix)
        test_sam = np.array(inv_scl).reshape(1 , 784)
        test_pred = clf.predict(test_sam)
        print(test_pred)
        cv2.imshow('frame' , gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

capture.release()
cv2.destroyAllWindows()