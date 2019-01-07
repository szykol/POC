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
# test_image = io.imread('test_rozdzielnie.jpg')
test_image = io.imread('test_lacznie.jpg')

display_img(test_image)

# przeksztalcenie obrazka do skali szarosci
grey = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
display_img(grey, 'W odcieniach szarosci')

# binaryzacja obrazka
# th = 100
th, test_bin = cv2.threshold(grey,0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
display_img(test_bin, 'Binarny')

#%%
import numpy as np
kernel = np.ones((10, 10),np.uint8)
# kernel = np.ones((3,3), np.uint8)
# erosion = cv2.erode(test_bin,kernel2,iterations = 3)
closing = cv2.morphologyEx(test_bin, cv2.MORPH_CLOSE, kernel, iterations=4)
cont_img = closing.copy()

display_img(closing)
_, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))

for cnt in contours:
    area = cv2.contourArea(cnt)
    # if area < 2000 or area > 4000:
    #     continue
    if len(cnt) < 5:
        continue
    ellipse = cv2.fitEllipse(cnt)
    # cv2.ellipse(test_image, ellipse, (0,255,0), 2)
    cv2.ellipse(test_image, ellipse, (0,255,0), 2, cv2.LINE_AA)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# closing = cv2.morphologyEx(test_bin,cv2.MORPH_OPEN,kernel)

display_img(test_image, 'Test')

#%%
# noise removal
kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(test_bin,cv2.MORPH_OPEN,kernel, iterations = 2)
opening = closing.copy()
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
display_img(dist_transform, 'Dist transform')
display_img(sure_fg, 'Sure fg')


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