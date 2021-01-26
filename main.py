import cv2
import os
import imutils as imutils
import numpy as np
import pytesseract
from skimage import img_as_bool, color, morphology
from skimage.morphology import skeletonize
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
tresholding = False  # progowanie
filename = ''


def detection(filename):
    watch_cascade = cv2.CascadeClassifier('Resources/cascade.xml')
    image = cv2.imread(filename)

    def detectPlateRough(image_gray,resize_h = 720,top_bottom_padding_rate = 0.05):
            if top_bottom_padding_rate>0.2:
                print("error:top_bottom_padding_rate > 0.2:",top_bottom_padding_rate)
                exit(1)
            height = image_gray.shape[0]
            padding = int(height*top_bottom_padding_rate)
            scale = image_gray.shape[1]/float(image_gray.shape[0])
            image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))
            #cv2.imshow('resized',image)
            image_color_cropped = image[padding:resize_h-padding,0:image_gray.shape[1]]
            image_gray = cv2.cvtColor(image_color_cropped,cv2.COLOR_RGB2GRAY)
            watches = watch_cascade.detectMultiScale(image_gray, scaleFactor=1.08, minSize=(36, 9),maxSize=(36*40, 80*40))
            # watches = watch_cascade.detectMultiScale(image_gray, scaleFactor=1.05, minNeighbors=5, minSize = (40,40))

            cropped_images = []
            plate_numbers= []
            plate_numbers_string = "Numery rejestracyjne: "

            for (x, y, w, h) in watches:

                # cv2.rectangle(image_color_cropped, (x, y), (x + w, y + h), (0, 0, 255), 1)

                x -= w * 0.14
                w += w * 0.28
                y -= h * 0.15
                h += h * 0.3

                #cv2.rectangle(image_color_cropped, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)

                cropped = cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
                cropped_images.append([cropped,[x, y+padding, w, h]])
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                imgResize = imutils.resize(gray, height=150)
                thresh = cv2.threshold(imgResize, 100, 255, cv2.THRESH_BINARY)[1]  # wszystkie wartości powyżej 100 zamieniane na 255(biały)
                output = image_to_string(thresh, lang='eng', config='--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPRSTQWYZ ')
                plate_numbers.append(output)

            if len(plate_numbers) == 0:
                plate_numbers_string = plate_numbers_string + "(nie wykryto tablic rejestracyjnych)"
            else:
                for plate in plate_numbers:
                    plate_numbers_string = plate_numbers_string + plate + ", "

            plate_numbers_string.replace("\n", " ")
            plate_numbers_string.replace("\\n", " ")
            print(plate_numbers_string)
            window["-SELECT_IMAGE_2-"].update(plate_numbers_string)

            return cropped_images

    def cropImage(image,rect):
            x, y, w, h = computeSafeRegion(image.shape,rect)
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

    images = []
    images = detectPlateRough(image,image.shape[0],top_bottom_padding_rate=0.1)
    print("Number of license plates detected", len(images))

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
    [sg.Button('Progowanie (binaryzacja)', key="-PROGOWANIE-")],
    [sg.Button('Filtry', key='-FILTERS-')], # Rozmycie Gaussa, Filtr Medianowy, Filtr wyostrzający
    [sg.Button('RGB -> HSV', key="-RGB_to_HSV-"), sg.Button('HSV -> RGB', key="-HSV_to_RBG-")],
    [sg.Button('Detekcja krawędzi', key="-EDGE_DETECTION-")],
    [sg.Button('Segmentacja', key="-SEGMENTATION-")],
    [sg.Button('Szkieletyzacja', key="-SKELETONIZATION-")],
    [sg.Button('Erozja i Dylatacja', key="-EROSION_DILATATION-")],
    [sg.Button('Reset',size=(25,1), key="-RESET-")]

]

center_column = [
    [sg.Text("Wybierz zdjęcie do analizy", key="-SELECT_IMAGE-")],
    [sg.Image(size=(40, 20),key="-IMAGE-")],
    [sg.Text(" ", size=(50, 3), font=("",14), key="-SELECT_IMAGE_2-")],
]

right_column = [
    [sg.Text("OPCJE", key="-OPTIONS_TEXT-", font=("",12))],
    [sg.HSeparator()],
    [sg.Button('Obróć w lewo', key="-ROTATE_LEFT-"), sg.Button('Obróć w prawo', key="-ROTATE_RIGHT-")],
    [sg.Button('Zmniejsz', key="-REDUCE_SIZE-"), sg.Button('Powiększ', key="-INCREASE_SIZE-")],
    [sg.Text('Jasność:'), sg.Button('+', key="-ADD_BRIGHTNESS-"), sg.Button('-', key="-UNDO_BRIGHTNESS-")],
    #[sg.Slider((1,100), key='_SLIDER_', orientation='h', enable_events=True, disable_number_display=True),
    #        sg.T('     ', key='_RIGHT_')],

]

# ----- Wypełnienie układu okna -----
layout = [
    [
        sg.Column(left_column),
        sg.VSeperator(),
        sg.Column(center_column, element_justification='c'),
        sg.VSeperator(),
        sg.Column(right_column, element_justification='c'),
    ]
]

window = sg.Window("Program", layout)

# Włączenie pętli zdarzeń
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        if(os.path.exists("Resources/Results/plik_roboczy.PNG")):
            os.remove("Resources/Results/plik_roboczy.PNG")
        if (os.path.exists("Resources/Results/negatyw.PNG")):
            os.remove("Resources/Results/negatyw.PNG")
        if (os.path.exists("Resources/Results/progowanie.PNG")):
            os.remove("Resources/Results/progowanie.PNG")
        if (os.path.exists("Resources/Results/skala_szarosci.PNG")):
            os.remove("Resources/Results/skala_szarosci.PNG")
        break

    if event == "-FOLDER-":
        try:
            filename = values["-FOLDER-"]
            originalFile = Image.open(filename)
            work_filename="Resources/Results/plik_roboczy.PNG"
            originalFile.save(work_filename)
            window["-IMAGE-"].update(filename=filename)
            window["-SELECT_IMAGE-"].update("")

            # Funkcja detekcji oraz OCR do znalezienia numeru rejestracyjnego
            detection(filename)
        except:
            pass

    if event == "-NEGATYW-":
        try:
            if(negative!=True):
                negative = True
                img = Image.open(work_filename)
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
                img.save("Resources/Results/negatyw.PNG")
                window["-IMAGE-"].update(filename="Resources/Results/negatyw.PNG")
            else:
                window["-IMAGE-"].update(filename=work_filename)
                negative = False
        except:
            pass

    if event == "-ODCIENIE_SZAROSCI-":
        try:
            if(gray!=True):
                gray = True
                image = cv2.imread(work_filename)
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("Resources/Results/skala_szarosci.PNG", img_gray)
                window["-IMAGE-"].update(filename="Resources/Results/skala_szarosci.PNG")
            else:
                window["-IMAGE-"].update(filename=work_filename)
                gray = False
        except:
            pass

    if event == "-NORMALIZACJA_HISTOGRAMU-":
        try:
            #Histogram
            img = cv2.imread(work_filename)
            plt.figure("Histogram")
            plt.hist(img.flatten(), 256, [0, 256], color='r')
            plt.xlim([0, 256])
            plt.title("Histogram")
            plt.legend(('histogram',), loc='upper right')

            # Znormalizowany Histogram
            img = cv2.imread(work_filename, 0)
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

    if event=="-PROGOWANIE-":
        try:
            if(tresholding!=True):
                tresholding = True
                image = cv2.imread(work_filename, 1)
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, img_bin1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
                cv2.imwrite("Resources/Results/progowanie.PNG", img_bin1)
                window["-IMAGE-"].update(filename="Resources/Results/progowanie.PNG")
            else:
                window["-IMAGE-"].update(filename=work_filename)
                tresholding = False
        except:
            pass

    if event == "-FILTERS-":
        try:
            image = cv2.imread(work_filename, 1)
            image = cv2.resize(image, (450, 400), None, .25, .25)

            gaussianBlurKernel = np.array(([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), np.float32)/9
            sharpenKernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            meanBlurKernel = np.ones((3, 3), np.float32)/9
            gaussianBlur = cv2.filter2D(src=image, kernel=gaussianBlurKernel, ddepth=-1)
            meanBlur = cv2.filter2D(src=image, kernel=meanBlurKernel, ddepth=-1)
            sharpen = cv2.filter2D(src=image, kernel=sharpenKernel, ddepth=-1)
            horizontalStack = np.concatenate((gaussianBlur, meanBlur, sharpen), axis=1)
            cv2.imshow("Filtr Gaussa, Filtr Medianowy, Filtr Wyostrzajacy", horizontalStack)
        except:
            pass

    if event=="-ROTATE_LEFT-":
        try:
            image = Image.open(work_filename)
            transposed = image.transpose(Image.ROTATE_90)
            transposed.save(work_filename)
            window["-IMAGE-"].update(filename=work_filename)
        except:
            pass

    if event=="-ROTATE_RIGHT-":
        try:
            image = Image.open(work_filename)
            transposed = image.transpose(Image.ROTATE_270)
            transposed.save(work_filename)
            window["-IMAGE-"].update(filename=work_filename)
        except:
            pass

    if event=="-REDUCE_SIZE-":
        try:
            image = cv2.imread(work_filename, 1)
            image_resized = cv2.resize(image, None, fx=0.5, fy=0.5)
            cv2.imwrite(work_filename, image_resized)
            # image_resized.save(work_filename)
            window["-IMAGE-"].update(filename=work_filename)
        except:
            pass

    if event=="-INCREASE_SIZE-":
        try:
            image = cv2.imread(work_filename, 1)
            height, width = image.shape[:2]
            image_resized = cv2.resize(image, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(work_filename, image_resized)

            window["-IMAGE-"].update(filename=work_filename)
        except:
            pass

    if event=="-RGB_to_HSV-":
        try:
            image = cv2.imread(work_filename, 1)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            cv2.imwrite(work_filename, image_hsv)
            window["-IMAGE-"].update(filename=work_filename)
        except:
            pass

    if event=="-HSV_to_RBG-":
        try:
            image = cv2.imread(work_filename, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            cv2.imwrite(work_filename, image_rgb)
            window["-IMAGE-"].update(filename=work_filename)
        except:
            pass

    if event=="-EDGE_DETECTION-":
        try:
            image = cv2.imread(work_filename, 0)
            edges = cv2.Canny(image, 100, 200)
            plt.figure(figsize=(10, 5))
            plt.subplot(121), plt.imshow(image, cmap='gray')
            plt.title('Oryginał'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(edges, cmap='gray')
            plt.title('Wykrycie krawędzi'), plt.xticks([]), plt.yticks([])
            plt.show()
        except:
            pass

    if event=="-SEGMENTATION-":
        try:
            img = cv2.imread(work_filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vectorized = img.reshape((-1, 3))
            vectorized = np.float32(vectorized)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            K = 3
            attempts = 10
            ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            result_image = res.reshape((img.shape))
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(img)
            plt.title('Zdjęcie oryginalne'), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 2, 2), plt.imshow(result_image)
            plt.title('Segmentacja zdjęcia K = %i' % K), plt.xticks([]), plt.yticks([])
            plt.show()
            window["-IMAGE-"].update(filename=work_filename)
        except:
            pass

    if event=="-SKELETONIZATION-":
        try:
            image = cv2.imread(work_filename, 1)
            image = color.rgb2gray(image)
            image = img_as_bool(image)

            image_skeletonized = skeletonize(image)
            plt.figure(figsize=(10, 5))
            plt.subplot(121), plt.imshow(image, cmap='gray', interpolation='nearest')
            plt.title('Format bool'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(image_skeletonized, cmap='gray', interpolation='nearest')
            plt.title('Szkieletyzacja'), plt.xticks([]), plt.yticks([])
            plt.show()
        except:
            pass

    if event=="-ADD_BRIGHTNESS-":
        try:
            image = cv2.imread(work_filename, 1)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            increase = 10
            v = hsv[:, :, 2]
            v = np.where((255-v)<increase,255,v+increase)
            hsv[:, :, 2] = v
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(work_filename, image)
            window['-IMAGE-'].update(filename=work_filename)
        except:
            pass

    if event=="-UNDO_BRIGHTNESS-":
        try:
            image = cv2.imread(work_filename, 1)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[...,2] = hsv[...,2]*0.5
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(work_filename, image)
            window['-IMAGE-'].update(filename=work_filename)
        except:
            pass

    if event=="-EROSION_DILATATION-":
        try:
            image = cv2.imread(work_filename, 1)
            kernel = np.ones((5, 5), np.uint8)
            image_erosion = cv2.erode(image, kernel, iterations=1)
            image_dilation = cv2.dilate(image, kernel, iterations=1)
            cv2.imshow('Erozja', image_erosion)
            cv2.imshow('Dylatacja', image_dilation)
        except:
            pass

    if event=="-RESET-":
        try:
            filename = values["-FOLDER-"]
            originalFile = Image.open(filename)
            originalFile.save(work_filename)
            window["-IMAGE-"].update(filename=filename)
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
