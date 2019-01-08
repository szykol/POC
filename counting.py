import matplotlib.pyplot as plt
from skimage import io
import scipy
from scipy import misc
import cv2


def display_img(im, title='default', cmap='gray'):
    """Funkcja wyświetlająca obrazek"""
    plt.figure(figsize=(15,15))
    plt.imshow(im, cmap=cmap)
    plt.title(title)
    plt.show()

def count_coins(im_url, tests=False, display_steps=False):
    """Funkcja przekształcająca obrazek i licząca ile monet znajduje sie na obrazku"""
    if not isinstance(im_url, str):
        raise TypeError(f'Nie podano prawidłowego typu danych. Oczekiwano (str), a otrzymano ({type(im_url)}).')

    if tests:
        display_steps = False
    # załadowanie oryginalnego obrazka
    # test_image = io.imread('test_rozdzielnie.jpg')
    #test_image = io.imread('img/Obraz (11).jpg')

    try:
        test_image = io.imread(im_url)
    except FileNotFoundError:
        raise ValueError('Nie podano prawidłowego pliku!')
    
    #test_image_small = cv2.resize(test_image,None,fx=0.2,fy=0.2)
    test_image_small = scipy.misc.imresize(test_image, 0.2)

    if not tests:
        display_img(test_image_small, 'Oryginał')

    # przeksztalcenie obrazka do skali szarosci
    shifted = cv2.pyrMeanShiftFiltering(test_image_small, 21, 51)
    grey = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)

    if display_steps:
        display_img(grey, 'W odcieniach szarosci')

    # binaryzacja obrazka
    # th = 100
    th, test_bin = cv2.threshold(grey,0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    if display_steps:
        display_img(test_bin, 'Binarny')

    # przeksztalcenie morfologiczne - zamkniecie
    import numpy as np
    kernel = np.ones((3, 3),np.uint8)
    closing = cv2.morphologyEx(test_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    if display_steps:
        display_img(closing)

    sure_bg = cv2.dilate(closing,kernel,iterations=3)
    # Znajdowanie pewnego pierwszego planu
    dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

    # Znajdowanie nieznanego regionu
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(test_image_small,markers)
    test_image_small[markers == -1] = [0,255,0]

    if display_steps:
        display_img(dist_transform, 'Dist transform')
        display_img(sure_fg, 'Sure fg')
    
    if not tests:
        display_img(test_image_small, 'Oryginał z wykrytymi monetami')

    # podliczanie obiektów
    from scipy import ndimage as ndi
    label_objects, nb_labels = ndi.label(sure_fg)

    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 20
    mask_sizes[0] = 0

    figures = mask_sizes[label_objects]

    # plt.imshow(figures, cmap="hot")
    # plt.axis('on')
    # plt.suptitle('Obiekty na obrazie')
    # plt.show()

    if not tests:
        display_img(figures, cmap="hot")

    print(f'Wykryto {nb_labels} obiekty/ów na obrazie')

    return nb_labels