import cv2
import numpy as np
import sys

from PIL import Image
import pytesseract

from matplotlib import pyplot as plt

# scale up dimensions (400 * 3, 130 * 3) aka BR plate dimensions * 3
plate_image_dimensions = (1200,390)

def lpr(image, show_steps = False, show_contour = True, show_plates = True, show_hist = True):
    colorimg = image
    image = cv2.cvtColor(colorimg,cv2.COLOR_BGR2GRAY)

    # pre-process the image
    ppimg = pre_process(image)

    # run through edge detection algorithm
    edgeimg = detect_edges(ppimg)

    if show_steps:
        helper_imshow("Original Image", colorimg)
        helper_imshow("Pre-Processed Image", ppimg)
        helper_imshow("Edge Detection Result", edgeimg)
        helper_imwait()

    # get the region where the license plate is
    (ok, out, approx, cnt) = try_get_license_plate(ppimg, edgeimg)


    # show steps if the flag is set
    if show_contour:
        tmp = cv2.cvtColor(ppimg.copy(), cv2.COLOR_GRAY2RGB)
        lst = [out, approx, cnt]
        cv2.drawContours(tmp, lst, 0, (255, 0, 0), 2)
        cv2.drawContours(tmp, lst, 1, (0, 255, 0), 2)
        cv2.drawContours(tmp, lst, 2, (0, 0, 255), 2)
        helper_imshow("License Plate Region", tmp)

        helper_imwait()

    if not ok:
        return "NOT FOUND"

    # get a separate, resized, license plate image
    plate, hist = separate_resize_plate(ppimg, out, approx, cnt, show_hist)

    # TODO: talvez dê pra desfocar a imagem antes do binário e mudar os thresholds
    # make the plate binary
    binplate, thresh = binarize_plate(plate, hist)

    # dilate and erode image to remove small letters and screws
    clearplate = remove_plate_details(binplate)

    # remove other smaller than the number components
    clearplate = plate_remove_nonconforming(clearplate)

    if show_hist and not show_plates:
        plt.plot(range(256), hist)
        plt.axvline(thresh)
        plt.show()

    if show_plates:
        helper_imshow("plate", plate)
        helper_imshow("binplate", binplate)
        helper_imshow("clearplate", clearplate)

        if show_hist:
            plt.plot(range(256), hist)
            plt.axvline(thresh)
            plt.show()

        helper_imwait()

    return "NOT AVAILABLE"

def try_get_license_plate(image, edgeimg):
    (ok, out, approx, cnt) = find_license_plate(edgeimg)

    if ok:
        return ok, out, approx, cnt

    _, otsuimg = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    helper_showwait("otsu", otsuimg)

    (ok, out, approx, cnt) = find_license_plate(otsuimg)

    if ok:
        return ok, out, approx, cnt

    adaptimg = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
            cv2.THRESH_BINARY, 35, 10)

    # adaptimg = cv2.bitwise_not(adaptimg)
    # adaptimg = cv2.dilate(adaptimg, np.ones((3,3), np.uint8), iterations = 1)
    # result = cv2.erode(result, np.ones((5,5), np.uint8), iterations = 2)

    helper_showwait("adapt", adaptimg)

    (ok, out, approx, cnt) = find_license_plate(adaptimg)

    if ok:
        return ok, out, approx, cnt

    return False, None, None, None

def separate_resize_plate(image, out, apr, cnt, show_hist = False):
    # Create an image containing only the plate
    cleanimg = image.copy()
    mask = np.full_like(cleanimg, 255)
    cv2.drawContours(mask, [cnt], 0, (0, 0, 0), -1)
    cv2.drawContours(mask, [cnt], 0, (255, 255, 255), 2)
    cleanimg = cv2.add(cleanimg, mask)

    # calculate histogram
    ri = cleanimg.ravel()
    rm = mask.ravel()
    hist = np.zeros(256)

    for i in range(len(rm)):
        if rm[i] == 0:
            hist[ri[i]] += 1

    # cumulative histogram
    cumulative = np.zeros_like(hist)
    cumulative[0] = hist[0]
    for i in range(len(cumulative) - 1):
        cumulative[i + 1] = cumulative[i] + hist[i + 1]

    pixels = cumulative[255]

    # equalized image creation
    (w,h) = cleanimg.shape
    clone = cleanimg.copy()
    for i in range(w):
        for j in range(h):
            if mask[i][j] == 0:
                clone[i][j] = np.int8((255 / pixels) * cumulative[cleanimg[i][j]])

    # work on the plate region
    (prx, pry, prw, prh) = cv2.boundingRect(apr)
    plate = cleanimg[pry:pry+prh, prx:prx+prw].copy()

    # Resize the plate
    plate = cv2.resize(plate, plate_image_dimensions)
    return plate, hist

def binarize_plate(plate, hist):
    thresh = calculate_otsu(hist)
    _, result = cv2.threshold(plate, thresh, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)
    return result, thresh

def calculate_otsu(hist): # adapted from wikipedia
    nbins = 256
    p = hist / np.sum(hist)
    sigma_b = np.zeros((256,1))
    for t in range(nbins):
        q_L = sum(p[:t])
        q_H = sum(p[t:])
        if q_L == 0 or q_H == 0:
            continue

        miu_L = sum(np.dot(p[:t], np.transpose(np.matrix([i for i in range(t)])))) / q_L
        miu_H = sum(np.dot(p[t:], np.transpose(np.matrix([i for i in range(t, 256)])))) / q_H
        sigma_b[t] = q_L * q_H * (miu_L - miu_H) ** 2

    return np.argmax(sigma_b)


# remove small details of a plate, as the city name and screws
def remove_plate_details(plate):
    # make small details dissapear
    result = cv2.dilate(plate, np.ones((5,5), np.uint8), iterations = 2)
    # those which weren't that small are back but there are less of them
    result = cv2.erode(result, np.ones((5,5), np.uint8), iterations = 2)
    return result

def plate_remove_nonconforming(plate):
    num_area_min, num_area_max = 5000, 50000
    inverted = cv2.bitwise_not(plate)
    img, contours, hi = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    curr_hi = hi[0]
    curr_cnt = 0
    while curr_cnt != -1:
        contour = contours[curr_cnt]

        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.05, True)
        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))

        bw, bh = helper_boxwh(box)
        ratio = bh/bw
        print(np.abs(cv2.contourArea(contour)), bw, bh, bw/bh)

        area = np.abs(cv2.contourArea(contour))
        if area < num_area_min or area > num_area_max or ratio < 1.20 or ratio > 6.90:
            cv2.drawContours(img, [contour], 0, (0,0,0), -1)

            tmpimg = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)

            # cv2.drawContours(tmpimg, [box, contour, approx], 0, (0,0,255), 2)
            # cv2.drawContours(tmpimg, [box, contour, approx], 1, (255,0,0), 2)
            # cv2.drawContours(tmpimg, [box, contour, approx], 2, (0,255,0), 2)
            # helper_showwait("CAIXA", tmpimg)

        curr_cnt = curr_hi[curr_cnt][0]

    return cv2.bitwise_not(inverted)


def pre_process(image):
    # enhance image contrast
    img = pp_enhance_contrast(image)

    # TODO: REMOVE AS NOT NEEDED, CANNY ALREADY DOES GAUSSIAN FILTERING
    # try and remove noise using a Gaussian Filter
    #img = cv2.GaussianBlur(img, (5,5), 0)
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
    # out = cv2.Canny(image, 50, 70, apertureSize=3, L2gradient=True)
    out = cv2.Canny(image, 50, 270, apertureSize=3, L2gradient=True)
    return out

def find_license_plate(image, accepted_ratio = 3.07, error = 0.37):
    area_threshold = 2000 # arbitrary threshold for plate area in image.
    img, contours, h = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(h)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.05, True)

        # find contours with 4 edges and the area of which is greater than threshold
        if len(approx) >= 4 and np.abs(cv2.contourArea(contour)) > area_threshold:
            rect = cv2.minAreaRect(contour)
            box = np.int0(cv2.boxPoints(rect))
            (box_w, box_h) = helper_boxwh(box)

            ratio = box_w / box_h

            # print(np.abs(cv2.contourArea(contour)), box_w, box_h, ratio)

            # tmpimg = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
            # cv2.drawContours(tmpimg, [box, contour, approx], 0, (0,0,255), 2)
            # cv2.drawContours(tmpimg, [box, contour, approx], 1, (255,0,0), 2)
            # cv2.drawContours(tmpimg, [box, contour, approx], 2, (0,255,0), 2)
            # helper_showwait("CAIXA", tmpimg)

            # brazilian license plate is 400mm x 130mm
            # 400 / 130 ~= 3.07... accept +- 0.3 error
            if accepted_ratio - error < ratio and ratio < accepted_ratio + error:
                # TODO: check approx is rectangle, if not "continue" the for loop

                # debug
                #cv2.drawContours(img, [box, contour, approx], -1, (0,0,255), 2)
                #cv2.drawContours(img, [box, contour, approx], 2, (0,255,0), 2)
                #helper_showwait("CAIXA", img)

                return (True, box, approx, contour)

    return False, None, None, None

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

    if np.abs(y2 - y1) < np.abs(y3 - y2):
        return (w,h)
    else:
        return (h,w)

# Code to initialize an image and print the license plate in it
if len(sys.argv) < 2:
    sys.exit("Usage: lpr.py <filename> [<show_steps>]")

image = cv2.imread(str(sys.argv[1]))

# TODO: remove // OR NOT (???), NORMALIZES IMAGE SIZES
image = cv2.resize(image, (1600, 900))

# Check the show steps argument
show_steps = False
if len(sys.argv) > 2:
    show_steps = (int(sys.argv[2]) == 1)

show_contour = False
if len(sys.argv) > 3:
    show_contour = (int(sys.argv[3]) == 1)

show_plates = False
if len(sys.argv) > 4:
    show_plates = (int(sys.argv[4]) == 1)

show_hist = False
if len(sys.argv) > 5:
    show_hist = (int(sys.argv[5]) == 1)

# print final result (plate's number)
print(lpr(image, show_steps, show_contour, show_plates, show_hist))
