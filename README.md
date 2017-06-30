# simple-alpr
A simple Automatic License Plate Recognition in Python using OpenCV and Tesseract.

Read the report PDF for more info. The image files are inside image.zip and are provided this way so that automatic data mining agorithms cant find license plates as easily.

In order to run this project you must have installed the libraries OpenCV (for python3), pytesseract (also for python3) and tesseract.

Basic usage is as follows:
`python lpr.py <image file.jpg> [<show steps> <show contours> <show plates> <show histogram>]`

where:
  <show steps> = 0 or 1: show original, pre processed and edge detected images
  <show contours> = 0 or 1: show the region where it detected a plate's contour
  <show plates> = 0 or 1: show the processed license plate region of interest
  <show histogram> = 0 or 1: shows the license plate ROI's histogram

by default all optional arguments are turned off.
