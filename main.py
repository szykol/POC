#%%
import matplotlib.pyplot as plt

def display_img(im, title='default', cmap='gray'):
    plt.figure(figsize=(15,15))
    plt.imshow(im, cmap=cmap)
    plt.title(title)
    plt.show()

#%%
from skimage import io
import cv2

# załadowanie oryginalnego obrazka
test_image = io.imread('test_rozdzielnie.jpg')

display_img(test_image)

# przeksztalcenie obrazka do skali szarosci
grey = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
display_img(grey, 'W odcieniach szarosci')

# binaryzacja obrazka
th = 100
th, test_bin = cv2.threshold(grey, thresh=th, maxval=255, type=cv2.THRESH_OTSU)
display_img(test_bin, 'Binarny')

#%%
import numpy as np
kernel = np.ones((35,35),np.uint8)
closing = cv2.morphologyEx(test_bin, cv2.MORPH_OPEN, kernel)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# closing = cv2.morphologyEx(test_bin,cv2.MORPH_OPEN,kernel)

display_img(closing, 'Test')
