import numpy as np
import cv2

d= "E:\\Dropbox (MIT)\\Tuomas\\Infant_results\\Specgrams\\Specgrams\\ph"

import os
i=0;

for root, dirs, files in os.walk(d):
	for file in files:
		if file.endswith("0.999.jpg"):
			print(os.path.join(root, file))
			i+=1
			img=cv2.imread(os.path.join(root, file))
			if i==1:
				cimage999=img
			else:
				cimage999 = np.concatenate((cimage999, img), axis=1)
			#height, width = cimage.shape[:2]
			#print(cimage.shape[:2])
if i>0:
	cv2.imwrite("out999.jpg",cimage999)
			
i=0			
for root, dirs, files in os.walk(d):
	for file in files:
		if file.endswith("0.998.jpg"):
			print(os.path.join(root, file))
			i+=1
			img=cv2.imread(os.path.join(root, file))
			if i==1:
				cimage998=img
			else:
				cimage998 = np.concatenate((cimage998, img), axis=1)
			#height, width = cimage.shape[:2]
			#print(cimage.shape[:2])			
			
if i>0:
	cv2.imwrite("out998.jpg",cimage998)

i=0
for root, dirs, files in os.walk(d):
	for file in files:
		if file.endswith("0.997.jpg"):
			print(os.path.join(root, file))
			i+=1
			img=cv2.imread(os.path.join(root, file))
			if i==1:
				cimage997=img
			else:
				cimage997 = np.concatenate((cimage997, img), axis=1)
			#height, width = cimage.shape[:2]
			#print(cimage.shape[:2])			
			
if i>0:
	cv2.imwrite("out997.jpg",cimage997)

i=0
for root, dirs, files in os.walk(d):
	for file in files:
		if file.endswith("0.996.jpg"):
			print(os.path.join(root, file))
			i+=1
			img=cv2.imread(os.path.join(root, file))
			if i==1:
				cimage996=img
			else:
				cimage996 = np.concatenate((cimage996, img), axis=1)
			#height, width = cimage.shape[:2]
			#print(cimage.shape[:2])			
	
if i>0:	
	cv2.imwrite("out996.jpg",cimage996)