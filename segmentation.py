# segmentacja
import numpy as np
from functools import partial
import cv2

image = np.array([
    [1,0,0,0,1,0,0,1],
    [1,0,0,1,1,0,0,1],
    [1,1,0,0,0,0,1,1],
    [1,1,0,0,1,1,1,0],
    [1,1,1,0,0,0,0,0],
    [0,0,1,0,0,0,1,0],
    [1,0,1,0,0,0,1,1],
    [0,0,1,0,0,0,1,0],
])

current_index = 0


im = cv2.imread('ksztalty.png')
# im = cv2.resize(im, (512,256)) 
grey = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
th, bin = cv2.threshold(grey,0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

def splitting(image):
    indices = image.copy()
    
    split(indices)     
    merge(image, indices)
    indices[image == 0] = 0
    new_indices = clear_indices(indices)

    new_image = color_objects(image, new_indices)
    # cv2.imwrite('test.png',new_image)

    print(image)
    print(indices)
    #             y,x
    print(indices[3,6])
    print(new_indices)
    return new_image

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
        return chunk
    else:
        current_index += 1
        return np.full((size_y, size_x), current_index)

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
                        print('Ustawiam(poprzedni)')
                        print(f'y: {y}, x: {index} last: {last}')
                        print(f'Ustawiam na {indices[y,last]}')
                    last = None

            directions = []
            if (y - 1) >= 0:
                directions.append((y-1, x))
            # if (x - 1) >= 0:
            #     directions.append((y, x - 1))
            # up = x, y-1
            # directions.append((x, y+1))
            # directions.append((x+1, y))

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
                        print('Ustawiam(sasiad)')
                        print(f'y: {y}, x: {index} last: {last}')
                        print(f'Ustawiam na {indices[close_set]}')
                    last = None
            
            elif last is None:
                # zapamietuje ten indeks, jesli juz nie jest zapamietany
                last = x

            # ustawiam index sasiadom
            # for d in directions:
            #     try:
            #         if not LUT[d] and image[d] == image[y,x]:
            #             indices[d] = indices[y,x]
            #             LUT[d] = True
            #     except IndexError:
            #         pass

            LUT[y,x] = True

    # print(indices)

def clear_indices(indices):
    copy_ind = indices.copy()


    index = 2
    while(True):
        new_tab = copy_ind[copy_ind > index]
        if(len(new_tab) == 0):
            print('Nie ma juz wiecej indeksow')
            break
        min_index = np.amin(new_tab)

        copy_ind[copy_ind == min_index] = index    
        index+=1

    return copy_ind

def color_objects(image, indices):
    max_index = np.amax(indices)
    # colored = np.zeros_like(image, dtype=tuple)
    colored = np.zeros((len(indices), len(indices[0]),3), np.uint8)
    colored[indices == 0] = (0, 0, 0)
    for i in range(1, max_index+1):
        colored[indices == i] = (13 * i, 25 * i, 30 * (max_index + 1 - i))
    
    return colored

# splitting(image)

cv2.imwrite('test.png',splitting(bin))
