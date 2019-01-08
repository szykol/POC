#%%
import matplotlib.pyplot as plt

def display_img(im, title='default', cmap='gray'):
    plt.figure(figsize=(15,15))
    plt.imshow(im, cmap=cmap)
    plt.title(title)
    plt.show()

#%%
from skimage import io
import skimage
import scipy
import cv2

# załadowanie oryginalnego obrazka
# test_image = io.imread('test_rozdzielnie.jpg')
#test_image = io.imread('img/Obraz (11).jpg')

test_image = io.imread('test_lacznie.jpg')

test_image_small = scipy.misc.imresize(test_image, 0.2)

display_img(test_image_small)

# przeksztalcenie obrazka do skali szarosci
shifted = cv2.pyrMeanShiftFiltering(test_image_small, 21, 51)
grey = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)
display_img(grey, 'W odcieniach szarosci')

# binaryzacja obrazka
# th = 100
th, test_bin = cv2.threshold(grey,0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
display_img(test_bin, 'Binarny')

#%%
import numpy as np
kernel = np.ones((3, 3),np.uint8)
closing = cv2.morphologyEx(test_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
display_img(closing)

#%%

sure_bg = cv2.dilate(closing,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,3)
ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(test_image_small,markers)
test_image_small[markers == -1] = [0,255,0]

display_img(dist_transform, 'Dist transform')
display_img(sure_fg, 'Sure fg')
display_img(test_image_small)

#%%
from scipy import ndimage as ndi
label_objects, nb_labels = ndi.label(sure_fg)

sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0

figures = mask_sizes[label_objects]

#%%
plt.imshow(figures, cmap="hot")
plt.axis('on')
plt.suptitle('Obiekty na obrazie')
plt.show()

print(figures.shape)
print(label_objects.shape)
print(nb_labels)