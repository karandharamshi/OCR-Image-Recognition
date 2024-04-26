from PIL import Image 			# create instances of this class
import pytesseract 			# import the necessary packages for the tesseract to work
import os 			        # import the necessary packages for os instances	
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
if cap.isOpened():
	_,frame=cap.read()
	cap.release()
	if _ and frame is not None:
		cv2.imwrite('img.jpg',frame)

image = cv2.imread('img.jpg')
grayimg = image
height, width, channels = image.shape

for i in range(height):
    for j in range(width):
        grayimg[i,j] = 0.3 * image[i,j][0] + 0.59 * image[i,j][1] +  0.11 * image[i,j][2]
cv2.imwrite('image.jpg',grayimg)


im = Image.open("image.jpg")
text = pytesseract.image_to_string(im, lang = 'eng')
print(text)
text_file = open("Output.txt", "w")
text_file.write("%s" % text)
text_file.close()
os.system("espeak -f Output.txt -s 120" )	# gives the output from the output.txt file by using the espeak command
print(text)					# prints the text in terminal

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "img.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
    text = eachObject["name"]
    text_file = open("Outputs.txt", "a")
    text_file.write("%s " % text)

text_file.close()
os.system("espeak -f Outputs.txt -s 120")




