from keras.models import load_model
import numpy as np
import cv2
import copy
import os

global classes
global count_data_collection
global rock_img 
global paper_img 
global scissor_img

def resize_img(image, size = 30):
	height, width = image.shape[:2]
	resized_img = cv2.resize(image, (int(size), int(size)), interpolation=cv2.INTER_AREA)
	return resized_img

def thresholdImage(img):
    img = cv2.GaussianBlur(img, (7,7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new

def computerPlay(pred):
	if 	pred == classes[2] or pred == classes[3]:
		cv2.imshow('I SAY.....', rock_img)
	elif pred == classes[0]:
		cv2.imshow('I SAY.....', none_img)
	elif pred == classes[1]:
		cv2.imshow('I SAY.....', paper_img)
	else:
		cv2.imshow('I SAY.....', scissor_img)

def playRPS(is_for_data_collection):
	global count_data_collection
	model = load_model('model_6cat_rps_1.h5')
	x0, y0, width = 10, 40, 300
	font = cv2.FONT_HERSHEY_SIMPLEX
	fx, fy = 10, 20
	cap = cv2.VideoCapture(0)
	
	while(True):
		ret, frame = cap.read()
		gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		gray = copy.deepcopy(gray_img)

		cv2.rectangle(gray, (x0,y0), (x0+width-1,y0+width-1), (100,0,250), 5)
		roi = gray[y0:y0+width,x0:x0+width]

		if is_for_data_collection:
			print count_data_collection
			count_data_collection += 1
			cv2.imshow('ROI Image', roi)
			file_name = "images/my_data/train/None/" + str(count_data_collection) + ".png"
			print file_name
			print cv2.imwrite(file_name, roi)
		else:
			cv2.putText(gray, 'Play Area: Make sure ur hands are here!', (fx,fy), font, 0.6, (0,250,0), 2, 1)
			#roi = thresholdImage(roi)
			roi = resize_img(roi)
			img = np.float32(roi) / 255
			img = np.expand_dims(img, axis=0)
			img = np.expand_dims(img, axis=-1)
			pred = classes[np.argmax(model.predict(img)[0])]
			print "Computer's prediction is " + pred
			computerPlay(pred)

		cv2.imshow('Original Image', gray)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	# constants
	classes = ['None','0','1','2','3','4','5', '6']
	img_path = 'rock_paper_scissors_image'
	rock_img = cv2.imread(img_path + '/rock.jpeg', cv2.IMREAD_GRAYSCALE)
	rock_img = resize_img(rock_img, 300)
	none_img = cv2.imread(img_path + '/none.jpg', cv2.IMREAD_GRAYSCALE)
	none_img = resize_img(none_img, 300)
	paper_img = cv2.imread(img_path + '/paper.jpg', cv2.IMREAD_GRAYSCALE)
	paper_img = resize_img(paper_img, 300)
	scissor_img = cv2.imread(img_path + '/scissor.jpg', cv2.IMREAD_GRAYSCALE)
	scissor_img = resize_img(scissor_img, 300)


	count_data_collection = 2588
	playRPS(False)