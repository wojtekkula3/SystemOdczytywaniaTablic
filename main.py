import cv2
import imutils as imutils
import numpy as np

plateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
img = cv2.imread("Resources/car2.png")
imgResize = cv2.resize(img, (620,480))

# imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
# imgEdge = cv2.Canny(imgGray, 30, 200)

# imgResize = imutils.resize(img, height=500)
# Ostrość
# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# imgSharp = cv2.filter2D(imgResize, -1, kernel)

plates = plateCascade.detectMultiScale(imgResize,1.1, 4)

for (x,y,w,h) in plates:
    cv2.rectangle(imgResize,(x,y),(x+w,y+h),(255,0,0),2)
    imgRoi = imgResize[y:y+h,x:x+w]
    imgResizeROI = imutils.resize(imgRoi, height=100)

    cv2.imshow("Region Of Interests", imgResizeROI)

print(imgResize.shape)

cv2.imshow("Test program", imgResize)
cv2.waitKey(0)
