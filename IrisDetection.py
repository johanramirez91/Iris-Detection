import cv2
import numpy as np
import glob
import math
from scipy.interpolate import interp1d
from PIL import Image

# Menú selección
print("Type '1' to start")
opc = int(input())

if opc == 1:
    ''' 
    load your eye image below: 
    '''
    img = cv2.imread("S1001R10.jpg")  # Read image

    imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    imgb = cv2.medianBlur(imgg, 5)  # smoth filter

    output = img.copy()
    output2 = img.copy()
    outputg = imgg.copy()
    outputg2 = imgg.copy()

    # Iris detection
    iris = cv2.HoughCircles(imgb, cv2.HOUGH_GRADIENT, 2, 120, param1=35, param2=60, minRadius=95, maxRadius=105)
    iris = np.uint16(np.around(iris))
    for i in iris[0, :]:
        # Draw iris circle
        cv2.circle(output2, (i[0], i[1]), i[2], (0, 127, 0), 2)
    ret, th1 = cv2.threshold(iris, 90, 240, cv2.THRESH_BINARY)

    # pupil detection
    circles = cv2.HoughCircles(imgb, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=70, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw pupil
        cv2.circle(output, (i[0], i[1]), i[2], (0, 127, 0), 2)

    # Cut iris area
    xi = int(iris[:, :, 0])
    yi = int(iris[:, :, 1])
    ri = int(iris[:, :, 2])
    crop = outputg2[yi - ri:yi + ri, xi - ri:xi + ri]

    # Remove pupil
    _, thresh = cv2.threshold(crop, 110, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=2)
    irisfinal = cv2.bitwise_and(crop, erosion)
    cv2.imshow("Detección iris y pupila", np.hstack([output2, output]))
    # cv2.imshow("Iris", irisfinal)

    # Normalize
    q = np.arange(0.00, np.pi * 2, 0.01)  # Theta parameter
    inn = np.arange(0, int(irisfinal.shape[0] / 2), 1)  # Radio
    cartisian_image = np.empty(shape=[inn.size, int(irisfinal.shape[1]), 3])
    m = interp1d([np.pi * 2, 0], [0, irisfinal.shape[1]])
    for r in inn:
        for t in q:
            polarX = int((r * np.cos(t)) + irisfinal.shape[1] / 2)
            polarY = int((r * np.sin(t)) + irisfinal.shape[0] / 2)
            cartisian_image[r][int(m(t) - 1)] = irisfinal[polarY][polarX]
    cartisian_image = cartisian_image.astype('uint8')
    imn = Image.fromarray(cartisian_image)
    imn.save('cartesian_eye.jpeg')
    # Matching
    diferencia = cv2.subtract(img, img)
    if not np.any(diferencia):
        print("Corresponde")
    else:
        print("Denegado")
    # cv2.imwrite("m10.png",irisfinal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Intente nuevamente")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
