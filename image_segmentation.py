import cv2
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import  train_test_split
from sklearn.multiclass import  OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
#Load train image

img_name='img.png'
# Enhancing Image Quality
def enhance_image(img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert the YUV image back to RGB format
        img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        #covert to hsv to adjust saturation
        img_hsv=cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(img_hsv)
        s+=10
        img_hsv=cv2.merge((h,s,v))
        img_output=cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img_output

# Train Function
def train():
        img = cv2.imread('./train/'+img_name, 1)
        try:
            if img==None:
                print("Can't find "+img_name+" in /train/ directory. Please add image and try again.")
                return
        except:
            pass
        # Load pre-processed Image
        img_load=enhance_image(img)
        cv2.imwrite('./train/enhanced/enhanced_'+img_name, img_load)
        # return
        #Resizing image to speed up training process
        img_load=cv2.resize(img_load,(0,0), fx=0.3, fy=0.3)
        # Conversion to HSV color space
        img_train=cv2.cvtColor(img_load, cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(img_train)
        # Selecting Vegitation Area (Classiftying pixels with hue value between 80 and 40 as vegetation.
        res=[]
        row=0
        for i in h:
            col = 0
            for value in i:
                if value<=80 and value>=40:  #90 and 30 for original image
                    res.append(1)
                else:
                    res.append(0)
                col += 1
            row+=1
        img_result=img_hsv=cv2.merge((h,s,v))
        train_set=[]
        # Creating a one dimensional train set from 2d img_train.
        for row in img_train:
            for col in row:
                train_set.append(col)

        train_set=np.asarray(train_set)
        res=np.asarray(res)
        print("Dimensions of loaded image: ", img_load.shape)
        print("Result: ", res.shape)
        img_output=cv2.cvtColor(img_result, cv2.COLOR_HSV2RGB)
        cv2.imwrite("Output.png",img_output)

        # Loading training data
        df_data=pd.DataFrame(train_set)
        X=train_set
        y=res
        # Splitting train data to test the accuracy of the model.
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=10)
        t0=time()

        # Grid Parameters
        param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
        # Support Vector Classifier from SVM
        model = GridSearchCV(SVC(), param_grid, verbose=3)
        # Training a model
        model.fit(X_train, y_train)
        t1=time()-t0
        print("Modeled trained in: {} seconds".format(round(t1,3)))
        # model best parameters and coeff
        print(model.best_estimator_)
        print(model.best_params_)
        # Prediction
        predictions=model.predict(X_test)
        # Check accuracy of a model
        print(classification_report(y_test, predictions))

        # Save Trained model to file
        joblib.dump(model, './trained_mode.sav')
        #

if __name__=="__main__":
        train()

