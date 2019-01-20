#%%
import matplotlib.pyplot as plt
from skimage import io
import scipy
from scipy import misc
import cv2

#%%
def display_img(im, title='default', cmap='gray'):
    """Funkcja wyświetlająca obrazek"""
    plt.figure(figsize=(15,15))
    plt.imshow(im, cmap=cmap)
    plt.title(title)
    plt.show()


def series_of_transformations(im, tests=False, display_steps=False):
    test_image_small = scipy.misc.imresize(im, 0.2)
    # test_image_small = cv2.resize(test_image, (450,230))

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
    # import np as np
    kernel = np.ones((5, 5),np.uint8)
    closing = cv2.morphologyEx(test_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    # kernel = np.ones((5,5),np.uint8)
    # erosion = cv2.erode(closing,kernel,iterations = 5)

    if display_steps:
        display_img(closing)
        # display_img(erosion)

    kernel = np.ones((5,5),np.uint8)
    sure_bg = cv2.dilate(closing,kernel,iterations=3)
    # Znajdowanie pewnego pierwszego planu
    dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

    # Znajdowanie nieznanego regionu
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    if display_steps:
        display_img(dist_transform, 'Dist transform')
        display_img(sure_fg, 'Sure fg')

    if not tests:
        # display_img(colored)
        display_img(test_image_small, 'Oryginał z wykrytymi monetami')

    return sure_fg



#%%
def count_coins(im_url, tests=False, display_steps=False):
    """Funkcja przekształcająca obrazek i licząca ile monet znajduje sie na obrazku"""
    if not isinstance(im_url, str):
        raise TypeError(f'Nie podano prawidłowego typu danych. Oczekiwano (str), a otrzymano ({type(im_url)}).')

    global current_index
    current_index = 0

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
    
    sure_fg = series_of_transformations(test_image, tests=tests, display_steps=display_steps)

    # count = splitting(sure_fg)
    indices = sure_fg.astype(int)
    # indices = sure_fg.copy()
    
    
    indices = split(indices)     
    # print(indices) 
    indices[sure_fg == 0] = -1
    merge(sure_fg, indices)
    # ustawianie tla na zerowy indeks

    # czyszczenie indeksow
    new_indices = clear_indices(indices)


    # new_image = color_objects(image, new_indices)
    
    count = len(np.unique(new_indices)) - 1
    for i in range(count + 1):
        color_index(sure_fg, new_indices, i)


    # a = np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
    unique, counts = np.unique(new_indices, return_counts=True)
    occurances = dict(zip(unique, counts))
    print(occurances)

    #{0: 7, 1: 4, 2: 1, 3: 2, 4: 1}

    # display_img(sure_fg)

    print(f'Wykryto {count} obiekty/ów na obrazie')

    return count


# segmentacja

#%%
import numpy as np
from functools import partial
import cv2
# import np as np

current_index = 0
#%%
def split(chunk, level=0):
    global current_index
    try:
        size_x = len(chunk[0])
    except IndexError:
        current_index += 1
        return current_index
    size_y = len(chunk)
    a = int(size_x / 2)
    b = int(size_y / 2)
    temp = True
    for i in range(size_y):
        for j in range(size_x):
            if(chunk[i,j] != chunk[0,0]):
                temp = False
                break
    
    if not temp:
        f = partial(split, level=level+1)
        chunk[:b,:a] = f(chunk[:b,:a])
        chunk[:b, a:size_x] = f(chunk[:b, a:size_x])
        chunk[b:size_y, a:size_x] = f(chunk[b:size_y, a:size_x])
        chunk[b:size_y, :a] = f(chunk[b:size_y, :a])
        # current_index += 1
        return chunk
    else:
        current_index += 1
        return np.full((size_y, size_x), current_index)

#%%
def merge(image, indices):
    LUT = np.zeros_like(image, bool)

    for y in range(len(indices)):
        # ostatni nie ustawiony
        last = None
        for x in range(len(indices[0])):

            if last is not None:
                if image[y, last] != image[y, x]:
                    # ustaw indeks wszystkich poprzednich
                    for index in range(x-1, last-1, -1):
                        indices[y,index] = indices[y, last]
                        # print('Ustawiam(poprzedni)')
                        # print(f'Ide z {x}')
                        # print(f'y: {y}, x: {index} last: {last}')
                        # print(f'Ustawiam na {indices[y,last]}')
                    last = None

            directions = []
            if (y - 1) >= 0:
                directions.append((y-1, x))

            close_set = None
            for d in directions:
                try: # jesli znalazlem sasiada -> zapamietuje
                    if LUT[d] and image[d] == image[y,x]:
                        close_set = d
                        break
                except IndexError:
                    pass
            
            if close_set is not None:
                # ustawiam index od sasiada
                indices[y,x] = indices[close_set]

                # ustawiam indeks na prawo
                try:
                    if(image[y, x+1] == image[y,x]):
                        indices[y,x+1] = indices[close_set]
                except IndexError:
                    pass

                # ustaw wszystkie poprzednie piksele
                # na ten indeks
                if last is not None:
                    for index in range(x-1, last-1, -1):
                        indices[y,index] = indices[close_set]
                        # print('Ustawiam(sasiad)')
                        # print(f'y: {y}, x: {index} last: {last}')
                        # print(f'Ustawiam na {indices[close_set]}')
                    last = None
            
            elif last is None:
                # zapamietuje ten indeks, jesli juz nie jest zapamietany
                last = x

            LUT[y,x] = True

    # print(indices)

#%%
def clear_indices(indices):
    copy_ind = indices.copy()

    index = 0
    while(True):
        new_tab = copy_ind[copy_ind > index]
        if(len(new_tab) == 0):
            # print('Nie ma juz wiecej indeksow')
            break
        min_index = np.amin(new_tab)

        copy_ind[copy_ind == min_index] = index    
        index+=1

    return copy_ind

#%%
def color_objects(image, indices):
    max_index = np.amax(indices)
    # colored = np.zeros_like(image, dtype=tuple)
    colored = np.zeros((len(indices), len(indices[0]),3), np.uint8)
    colored[indices == 0] = (0, 0, 0)
    for i in range(1, max_index+1):
        colored[indices == i] = (13 * i, 25 * i, 30 * (max_index + 1 - i))
    
    return colored

#%%
def color_index(image, indices, index):
    colored = np.zeros((len(indices), len(indices[0]),3), np.uint8)
    colored[indices==index] = (255, 0, 255)
    # display_img(colored, f'obiekt nr. {index}')


def get_coin_size_in_pixels(im_url):
    """ Zwraca wielkość monety w pikselach """
    
    if not isinstance(im_url, str):
        raise TypeError(f'Nie podano prawidłowego typu danych. Oczekiwano (str), a otrzymano ({type(im_url)}).')

    try:
        test_image = io.imread(im_url)
    except FileNotFoundError:
        raise ValueError('Nie podano prawidłowego pliku!')
    
    coin_im = series_of_transformations(test_image, tests=True)
    display_img(coin_im)

    indices = coin_im.astype(int)
    # indices = sure_fg.copy()
    
    
    indices = split(indices)     
    # print(indices) 
    indices[coin_im == 0] = -1
    merge(coin_im, indices)
    # ustawianie tla na zerowy indeks

    # czyszczenie indeksow
    new_indices = clear_indices(indices)

    # a = np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
    unique, counts = np.unique(new_indices, return_counts=True)
    occurances = dict(zip(unique, counts))
    print(occurances)

    return occurances[0]



#%%
# current_index = 0
# count_coins('img/monety2.jpg',   display_steps=False)
# count_coins('img/monety14.jpg',   display_steps=False)
count_coins('img/monety16.jpg',   tests=True)
print(f'Rozmiar piątaka: {get_coin_size_in_pixels("img/5.jpg")}')
# count_coins('img/monety11.jpg',  display_steps=False)
# count_coins('img/monety13.jpg',  display_steps=False)
# count_coins('img/monety14.jpg',  display_steps=False)

# for i in range(1, 16):
#     count_coins(f'img/monety{i}.jpg', tests=True)
