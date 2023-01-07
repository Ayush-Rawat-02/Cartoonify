import cv2
import sys
import numpy as np

def cartoon(image):
	if image is None:
		sys.exit("Could not read image")
	cv2.imshow('Custom Image', image)
	k=cv2.waitKey(0)
	if k==ord('s'):
		cv2.imwrite("Test1.jpg", img)
	return image

img=cv2.imread(cv2.samples.findFile("Image1(walking).jpg"))
#cartoon(img)


# creating a grayscaled edge mask
def edges(img, thickness, blur_value):
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Grayscale image', grayImg)
	#k=cv2.waitKey(0)
	
	#   adding noise to image(blur effect)
	grayBlurred = cv2.medianBlur(grayImg, blur_value)
	#cv2.imshow('Blurred gray image', grayBlurred)
	#k=cv2.waitKey(0)
	
	#extracting edges using adaptive threshold
	edgeLayer = cv2.adaptiveThreshold(grayBlurred,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,thickness,blur_value)
	#cv2.imshow('Edges of image', edgeLayer)
	#k=cv2.waitKey(0)
	return edgeLayer
	

#edges(img,7,5)


#Color quantization(reducing the number of colors using k-means clustering
#k is the number of clusters as well as number of colors in the result
def quantize(img, k):
	#image transformation
	data = np.float32(img).reshape((-1,3))
	#determine criteria
	criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
	#implementing k means clustering
	ret, label, center = cv2.kmeans( data, k, None, criteria, 10,cv2.KMEANS_RANDOM_CENTERS)
	
	center = np.uint8(center)
	
	result = center[label.flatten()]
	result = result.reshape(img.shape)
	#cv2.imshow(' Quantised image layer', result)
	#k=cv2.waitKey(0)
	return result
	
quantized_layer = quantize(img, 4)

#reduce the noise after quantization

Quantized_layer = cv2.bilateralFilter(quantized_layer, d=7, sigmaColor=200, sigmaSpace=200)
#cv2.imshow(' Quantised image layer 2', quantized_mask)
#k=cv2.waitKey(0)
edgeLayer = edges(img,7,5)

#merging the edge mask with quantized layer using bitwise and
CartoonImage = cv2.bitwise_and(Quantized_layer, Quantized_layer, mask=edgeLayer)
cv2.imshow('Cartoonified image', CartoonImage)
k=cv2.waitKey(0)


#Video processing
"""
cap = cv2.VideoCapture('video.mp4')
if not cap.isOpened():
    	print("Cannot read video file")
    	exit()
while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		print("Cannot recieve frame (stream end?). Exiting...")
		break
	colored = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame', colored)
	if cv2.waitKey(1)==ord('q'):
		break;
cap.release()
cv2.destroyAllWindows()
"""
