import cv2
import numpy as np
import sys
from PIL import Image
import pytesseract

def lpr(image, show_steps = False):
	ppimg = pre_process(image)
	edgeimg = detect_edges(ppimg)
	(out, approx, cnt) = find_license_plate(edgeimg)

	(prx, pry, prw, prh) = cv2.boundingRect(approx)
	plate = ppimg[prx:prx+prw, pry:pry+prh].copy()

	if show_steps:
		helper_imshow("Original Image", image)
		helper_imshow("Pre-Processed Image", ppimg)
		helper_imshow("Edge Detection Result", edgeimg)

		tmp = cv2.cvtColor(ppimg.copy(), cv2.COLOR_GRAY2RGB)
		lst = [out, approx, cnt]
		cv2.drawContours(tmp, lst, 0, (255, 0, 0), 2)
		cv2.drawContours(tmp, lst, 1, (0, 255, 0), 2)
		cv2.drawContours(tmp, lst, 2, (0, 0, 255), 2)
		helper_imshow("License Plate Region", tmp)

		helper_imwait()

	return "NOT AVAILABLE"

def pre_process(image):
	# enhance image contrast
	img = pp_enhance_contrast(image)

	# try and remove noise using a Gaussian Filter
	#img = cv2.GaussianBlur(img, (5,5), 0)

	# show images for testing purposes
	#helper_imshow("PP: Contrast-Enhanced", img)
	#helper_imshow("PP: Gaussian Filter", img)
	#helper_imwait()
	return img

def pp_enhance_contrast(image):
	out_min = 2
	out_max = 255
	(in_min, in_max, _, _) = cv2.minMaxLoc(image)
	c_factor = (out_max - out_min) / (in_max - in_min)
	return ((image - in_min) * c_factor + out_min).astype(np.uint8)

def detect_edges(image):
	# use Canny edge detection, which is an application of
	# Sobel operators on the image followed by an algorithm
	# to better filter which (supposed) edges are "actually" edges.
	# The output is a binary image.
	out = cv2.Canny(image, 50, 70)
	return out

def find_license_plate(image):
	area_threshold = 2000
	img, contours, h = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:
		approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.05, True)

		if len(approx) == 4 and np.abs(cv2.contourArea(contour)) > area_threshold:
			rect = cv2.minAreaRect(contour)
			box = np.int0(cv2.boxPoints(rect))
			(box_w, box_h) = helper_boxwh(box)

			ratio = box_w / box_h

			# 400 / 130 ~= 3.07... accept +- 0.3 error
			if 2.7 < ratio and ratio < 3.4:
				# TODO: check approx

				# debug
				#cv2.drawContours(img, [box, contour, approx], -1, (0,0,255), 2)
				#cv2.drawContours(img, [box, contour, approx], 2, (0,255,0), 2)
				#helper_showwait("CAIXA", img)

				return (box, approx, contour)

	sys.exit("No license plate found.")

def helper_imshow(name, image):
	cv2.imshow(name, image)

def helper_imwait():
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def helper_showwait(name, image):
	helper_imshow(name, image)
	helper_imwait()

def helper_boxwh(box):
	x1 = box[0][0]
	y1 = box[0][1]
	x2 = box[1][0]
	y2 = box[1][1]
	x3 = box[2][0]
	y3 = box[2][1]

	w = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
	h = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

	if w < h:
		tmp = h
		h = w
		w = tmp

	return (w,h)

# Code to initialize an image and print the license plate in it
if len(sys.argv) < 2:
	sys.exit("Usage: lpr.py <filename> [<show_steps>]")

image = cv2.imread(str(sys.argv[1]), cv2.IMREAD_GRAYSCALE)

# TODO: remove
image = cv2.resize(image, (1280, 720))

# Check the show steps argument
show_steps = False
if len(sys.argv) > 2:
	show_steps = (int(sys.argv[2]) == 1)

print(lpr(image, show_steps))