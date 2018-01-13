import cv2
from image_segmentation import enhance_image
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

file_name='test.tif'


def test(_file_name):
        img_load=cv2.imread('./test/'+_file_name, 1)
        try:
            if img_load==None:
                print("Can't find "+file_name+" in /train/ directory. Please add image and try again.")
                return
        except:
            pass
        # Preprocessed
        print("Testing the image.")
        img=enhance_image(img_load)
        # Saving pre-processed image
        cv2.imwrite('./test/enhanced/enhanced_'+_file_name,img )
        # Conversion to HSV color space and creating a test_set from 2d image
        img_test=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        img_load = cv2.cvtColor(img_load, cv2.COLOR_BGR2HSV)
        test_set=[]
        test_img = []
        for row in img_test:
            for col in row:
                test_set.append(col)


        for row in img_lnoad:
            for col in row:
                test_img.append(col)

        test_set=np.asarray(test_set)
        test_img = np.asarray(test_img)
        # Original Image

        # LoadModel and predict result
        model=joblib.load('./trained_mode.sav')
        result=model.predict(test_set)
        row=0
        for i in result:
            if i==1:
                test_img[row]=[60,255,255]

            row+=1
        result_img=test_img.reshape(img.shape)
        result_img=cv2.cvtColor(result_img, cv2.COLOR_HSV2BGR)
        cv2.imshow('Output',result_img)
        cv2.imwrite('./result/result_'+file_name,result_img )
        cv2.waitKey(0)
        print(result)



if __name__=='__main__':
    test(file_name)