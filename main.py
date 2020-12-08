import cv2
import imutils as imutils
import numpy as np

plateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
img = cv2.imread("Resources/car2.png")
# img = cv2.imread("Resources/car1.jpg")
imgResize = cv2.resize(img, (620,480))

# Zmień na skalę szarości
img_gray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)

# Usuń szumy
img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)
cv2.imshow("Noise", img_gray);

# Krawędzie Canny
img_edge = cv2.Canny(img_gray, 130, 200)
cv2.imshow("Edge 2", img_edge);

# Znajdz kontury w opraciu o krawędzie
contours, hierarch = cv2.findContours(img_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

for contour in contours:
    parameter = cv2.arcLength(contour, True)
    approx= cv2.approxPolyDP(contour, 0.02 * parameter, True)

    if len(approx) == 4:
        contour_with_license_plate = approx
        x, y, w, h = cv2.boundingRect(contour)
        license_plate = img_gray[y:y+h, x:x+w]
        break

cv2.imshow("License plate", license_plate)


# imgResize = imutils.resize(img, height=500)
# Ostrość
# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# imgSharp = cv2.filter2D(imgResize, -1, kernel)

# plates = plateCascade.detectMultiScale(imgResize, 1.1, 4)
#
# for (x,y,w,h) in plates:
#     cv2.rectangle(imgResize,(x,y),(x+w,y+h),(255,0,0),2)
#     imgRoi = imgResize[y:y+h,x:x+w]
#     imgResizeROI = imutils.resize(imgRoi, height=100)
#     cv2.imshow("Region Of Interests", imgResizeROI)
#
# print(imgResize.shape)
# cv2.imshow("Photo", imgResize)
cv2.waitKey(0)
