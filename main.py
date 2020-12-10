import cv2
import imutils as imutils
import numpy as np
import pytesseract
from pytesseract import image_to_string
pytesseract.pytesseract.tesseract_cmd = "Resources/Tesseract-OCR/tesseract.exe"
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageFilter
import io
import os.path

sg.theme('DarkAmber')
negative = False
gray = False


def detection(filename):
    watch_cascade = cv2.CascadeClassifier('Resources/cascade.xml')
    image = cv2.imread(filename)

    def detectPlateRough(image_gray,resize_h = 720,en_scale =1.08 ,top_bottom_padding_rate = 0.05):
            if top_bottom_padding_rate>0.2:
                print("error:top_bottom_padding_rate > 0.2:",top_bottom_padding_rate)
                exit(1)
            height = image_gray.shape[0]
            padding = int(height*top_bottom_padding_rate)
            scale = image_gray.shape[1]/float(image_gray.shape[0])
            image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))
            image_color_cropped = image[padding:resize_h-padding,0:image_gray.shape[1]]
            image_gray = cv2.cvtColor(image_color_cropped,cv2.COLOR_RGB2GRAY)
            watches = watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),maxSize=(36*40, 80*40))
            # watches = watch_cascade.detectMultiScale(image_gray, scaleFactor=1.05, minNeighbors=5, minSize = (40,40))

            cropped_images = []
            print("out")
            for (x, y, w, h) in watches:
                print("in")

                #cv2.rectangle(image_color_cropped, (x, y), (x + w, y + h), (0, 0, 255), 1)

                x -= w * 0.14
                w += w * 0.28
                y -= h * 0.15
                h += h * 0.3

                #cv2.rectangle(image_color_cropped, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)

                cropped = cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
                cropped_images.append([cropped,[x, y+padding, w, h]])
                # cv2.imshow("CimageShow", cropped)

                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  # konwersja na czarno białe
                imgResize = imutils.resize(gray, height=150)
                thresh = cv2.threshold(imgResize, 100, 255, cv2.THRESH_BINARY)[1]  # wszystkie wartości powyżej 127 zamieniane na 255(biały)
                # cv2.imshow("grey", thresh)
                output = image_to_string(thresh, lang='eng', config='--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPRSTQWYZ ')
                print('Output: ', output)
                out = "Numery rejestracyjne pojazdów: "+output
                window["-SELECT_IMAGE_2-"].update(out)

                # cv2.waitKey(0)
                # plt.show()  # wyświetlanie obrazka

                # cv2.waitKey(0)
            return cropped_images

    def cropImage(image,rect):
            # cv2.imshow("imageShow", image)
            # cv2.waitKey(0)
            x, y, w, h = computeSafeRegion(image.shape,rect)
            # cv2.imshow("imageShow", image[y:y+h,x:x+w])
            # cv2.waitKey(0)
            return image[y:y+h,x:x+w]


    def computeSafeRegion(shape,bounding_rect):
            top = bounding_rect[1] # y
            bottom  = bounding_rect[1] + bounding_rect[3] # y +  h
            left = bounding_rect[0] # x
            right =   bounding_rect[0] + bounding_rect[2] # x +  w
            min_top = 0
            max_bottom = shape[0]
            min_left = 0
            max_right = shape[1]

            #print(left,top,right,bottom)
            #print(max_bottom,max_right)

            if top < min_top:
                top = min_top
            if left < min_left:
                left = min_left
            if bottom > max_bottom:
                bottom = max_bottom
            if right > max_right:
                right = max_right
            return [left,top,right-left,bottom-top]

    images = detectPlateRough(image,image.shape[0],top_bottom_padding_rate=0.1)

# Podział okna na dwie kolumny

left_column = [
    [
        sg.Text("Wybierz:"),
        sg.In(size=(30, 1), enable_events=True, key="-FOLDER-"),
        sg.FileBrowse("Szukaj"),
    ],
    [sg.Text("")],
    [sg.Button('Negatyw Obrazu', key="-NEGATYW-")],
    [sg.Button('Konwersja do odcieni szarości', key="-ODCIENIE_SZAROSCI-")],
    [sg.Button('Normalizacja histogramu', key="-NORMALIZACJA_HISTOGRAMU-")],
    [sg.Button('? Skalowanie')],
    [sg.Button('Progowanie (binaryzacja)')],
    [sg.Button('? Filtry (3 do wyboru)')],
    [sg.Button('Transformacja między przestrzeniami barw')],
    [sg.Button('Obrót')],
    [sg.Button('Zmiana jasności')],
    [sg.Button('Detekcja krawędzi')],
    [sg.Button('Segmentacja (3 metody)')],
    [sg.Button('Szkieletyzacja')],
    [sg.Button('Erozja/Dylatacja')],
    [sg.Button('Implementacja algorytmu OCR')],
    [sg.Button('Klasyfikator cech')]

]

right_column = [
    [sg.Text("Wybierz zdjęcie do analizy", key="-SELECT_IMAGE-")],
    [sg.Image(size=(40, 20),key="-IMAGE-")],
    [sg.Text(" ", size=(50, 1), font=("",14), key="-SELECT_IMAGE_2-")],
]

# ----- Wypełnienie układu okna -----
layout = [
    [
        sg.Column(left_column),
        sg.VSeperator(),
        sg.Column(right_column, element_justification='c'),
    ]
]

window = sg.Window("Program", layout)

# Włączenie pętli zdarzeń
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        try:
            filename = values["-FOLDER-"]
            window["-IMAGE-"].update(filename=filename)
            window["-SELECT_IMAGE-"].update("")
            # Funkcja detekcji oraz OCR do znalezienia numeru rejestraci
            detection(filename)
        except:
            pass

    if event == "-NEGATYW-":
        try:
            if(negative!=True):
                negative = True
                img = Image.open(filename)
                for i in range(0, img.size[0] - 1):
                    for j in range(0, img.size[1] - 1):
                        # Get pixel value at (x,y) position of the image
                        pixelColorVals = img.getpixel((i, j));
                        # Invert color
                        redPixel = 255 - pixelColorVals[0];  # Negate red pixel
                        greenPixel = 255 - pixelColorVals[1];  # Negate green pixel
                        bluePixel = 255 - pixelColorVals[2];  # Negate blue pixel
                        # Modify the image with the inverted pixel values
                        img.putpixel((i, j), (redPixel, greenPixel, bluePixel));
                img.save("Resources/Results/negatyw.PNG")  # write out the image as .png
                window["-IMAGE-"].update(filename="Resources/Results/negatyw.PNG")
            else:
                window["-IMAGE-"].update(filename=filename)
                negative = False
        except:
            pass

    if event == "-ODCIENIE_SZAROSCI-":
        try:
            if(gray!=True):
                gray = True
                image = cv2.imread(filename)
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("Resources/Results/skala_szarosci.PNG", img_gray) # write out the image as .png
                window["-IMAGE-"].update(filename="Resources/Results/skala_szarosci.PNG")
            else:
                window["-IMAGE-"].update(filename=filename)
                gray = False
        except:
            pass

    if event == "-NORMALIZACJA_HISTOGRAMU-":
        window["-IMAGE-"].update(filename=filename)
        try:
            img = cv2.imread(filename)
            plt.figure("Histogram")
            plt.hist(img.flatten(), 256, [0, 256], color='r')
            plt.xlim([0, 256])
            plt.title("Histogram")
            plt.legend(('histogram',), loc='upper right')

            img = cv2.imread(filename, 0)
            plt.figure("Znormalizowany histogram")
            img_normalized = cv2.equalizeHist(img)
            plt.hist(img_normalized.flatten(), 256, [0, 256], color='r')
            plt.xlim([0, 256])
            plt.title("Histogram znormalizowany")
            plt.legend(('histogram',), loc='upper right')

            merge_img = np.hstack((img, img_normalized))  # stacking images side-by-side
            cv2.imshow('Normalizacja zdjecia', merge_img)
            plt.show()
            cv2.waitKey(0)
        except:
            pass

window.close()



# Działajaca detekcja
# watch_cascade = cv2.CascadeClassifier('Resources/cascade.xml')
# image = cv2.imread("Resources/car2.png")
#
# def detectPlateRough(image_gray,resize_h = 720,en_scale =1.08 ,top_bottom_padding_rate = 0.05):
#         if top_bottom_padding_rate>0.2:
#             print("error:top_bottom_padding_rate > 0.2:",top_bottom_padding_rate)
#             exit(1)
#         height = image_gray.shape[0]
#         padding = int(height*top_bottom_padding_rate)
#         scale = image_gray.shape[1]/float(image_gray.shape[0])
#         image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))
#         image_color_cropped = image[padding:resize_h-padding,0:image_gray.shape[1]]
#         image_gray = cv2.cvtColor(image_color_cropped,cv2.COLOR_RGB2GRAY)
#         watches = watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),maxSize=(36*40, 9*40))
#         cropped_images = []
#         for (x, y, w, h) in watches:
#
#             #cv2.rectangle(image_color_cropped, (x, y), (x + w, y + h), (0, 0, 255), 1)
#
#             x -= w * 0.14
#             w += w * 0.28
#             y -= h * 0.15
#             h += h * 0.3
#
#             #cv2.rectangle(image_color_cropped, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
#
#             cropped = cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
#             cropped_images.append([cropped,[x, y+padding, w, h]])
#             # cv2.imshow("CimageShow", cropped)
#
#             gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  # konwersja na czarno białe
#             imgResize = imutils.resize(gray, height=150)
#             thresh = cv2.threshold(imgResize, 100, 255, cv2.THRESH_BINARY)[1]  # wszystkie wartości powyżej 127 zamieniane na 255(biały)
#             cv2.imshow("grey", thresh)
#             output = image_to_string(thresh, lang='eng', config='--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPRSTQWYZ')
#             print('Output: ', output)
#             cv2.waitKey(0)
#             # plt.show()  # wyświetlanie obrazka
#
#             # cv2.waitKey(0)
#         return cropped_images
#
# def cropImage(image,rect):
#         cv2.imshow("imageShow", image)
#         cv2.waitKey(0)
#         x, y, w, h = computeSafeRegion(image.shape,rect)
#         cv2.imshow("imageShow", image[y:y+h,x:x+w])
#         cv2.waitKey(0)
#         return image[y:y+h,x:x+w]
#
#
# def computeSafeRegion(shape,bounding_rect):
#         top = bounding_rect[1] # y
#         bottom  = bounding_rect[1] + bounding_rect[3] # y +  h
#         left = bounding_rect[0] # x
#         right =   bounding_rect[0] + bounding_rect[2] # x +  w
#         min_top = 0
#         max_bottom = shape[0]
#         min_left = 0
#         max_right = shape[1]
#
#         #print(left,top,right,bottom)
#         #print(max_bottom,max_right)
#
#         if top < min_top:
#             top = min_top
#         if left < min_left:
#             left = min_left
#         if bottom > max_bottom:
#             bottom = max_bottom
#         if right > max_right:
#             right = max_right
#         return [left,top,right-left,bottom-top]
#
# images = detectPlateRough(image,image.shape[0],top_bottom_padding_rate=0.1)






# plateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
# img = cv2.imread("Resources/car2.png")
# # img = cv2.imread("Resources/car1.jpg")
# imgResize = cv2.resize(img, (620,480))
#
# # Zmień na skalę szarości
# img_gray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
#
# # Usuń szumy
# img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)
# cv2.imshow("Noise", img_gray);
#
# # Krawędzie Canny
# img_edge = cv2.Canny(img_gray, 130, 200)
# cv2.imshow("Edge 2", img_edge);
#
# # Znajdz kontury w opraciu o krawędzie
# contours, hierarch = cv2.findContours(img_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
#
# for contour in contours:
#     parameter = cv2.arcLength(contour, True)
#     approx= cv2.approxPolyDP(contour, 0.02 * parameter, True)
#
#     if len(approx) == 4:
#         contour_with_license_plate = approx
#         x, y, w, h = cv2.boundingRect(contour)
#         license_plate = img_gray[y:y+h, x:x+w]
#         break
#
# cv2.imshow("License plate", license_plate)



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
# cv2.waitKey(0)
