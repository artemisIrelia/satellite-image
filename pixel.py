import cv2
from image_segmentation import enhance_image
img_name='img1asdf.png'
pixel=cv2.imread('./train/'+img_name,1)
if pixel==None:
	print("Image not found")
	exit
enhanced_image=enhance_image(pixel)
cv2.imwrite('./enhanced/enhanced_'+img_name, enhanced_image)