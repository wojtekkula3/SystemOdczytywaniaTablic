import cv2
import imutils as imutils
import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageFilter
import io
import os.path

sg.theme('DarkAmber')
negative = False
gray = False

# First the window layout in 2 columns

file_list_column = [
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

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Wybierz zdjęcie do analizy", key="-SELECT_IMAGE-")],
#   [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(size=(40, 20),key="-IMAGE-")],
    [sg.Text(" ", size=(40, 5), font=("",15), key="-SELECT_IMAGE_2-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column, element_justification='c'),
    ]
]

window = sg.Window("Program", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        try:
            filename = values["-FOLDER-"]
#           window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)
            window["-SELECT_IMAGE-"].update("")
            window["-SELECT_IMAGE_2-"].update("Numer rejestracyjny pojazdu: ")
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
                img.save("negatyw.PNG")  # write out the image as .png
                window["-IMAGE-"].update(filename="negatyw.PNG")
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
                print(image)
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                print(img_gray)
                cv2.imwrite("skala_szarosci.PNG", img_gray) # write out the image as .png
                window["-IMAGE-"].update(filename="skala_szarosci.PNG")
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
