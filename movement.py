import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
import cv2

def compare_frames(prev, next, draw_rectangles=False):
    diff = cv2.absdiff(prev, next)

    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    ret, diff2 = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(diff2, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    minArea = 50
    margin = 3

    next_copy = next.copy()

    coords = []
    for c in cnts:
        if cv2.contourArea(c) > minArea:
            rect = cv2.boundingRect(c)
            (x, y, w, h) = rect
            other_rect = x - margin, y - margin, w + margin, h + margin
            # print('==Start==')
            # print(rect)
            # print(find_middle(rect))
            # print('==End==')

            if draw_rectangles:
                cv2.rectangle(next_copy, (x - margin, y - margin), (x + w + margin, y + h + margin), (0, 0, 255), 2)
            coords.append(find_middle(rect))

    next_copy = apply_markers(next_copy, coords)
    # rect = (0,0,1280,720)q
    # middle = find_middle((0,0,1280,720))
    # next_copy = apply_markers(next_copy, [middle])

    return next_copy

def apply_markers(im, coords, size=15):

    image = np.copy(im)
    
    for item in coords:
        x, y = item
        for j in range(y-size, y+size+1):
            try:
                image[j,x] = (0, 0, 255)
            except IndexError:
                pass
        for i in range(x-size, x+size+1):
            try:
                image[y,i] = (0, 0, 255)
            except IndexError:
                pass
            
    return image

def find_middle(rect):
    """ Znajduje środek prostopadłościanu """
    (x, y, w, h) = rect
    return int(x + (w / 2)), int(y + (h / 2))



cam = True
cap = cv2.VideoCapture('examples/ball.mp4' if not cam else 0)

prev_frame = None

if not cap.isOpened():
    print('Filmik nie działa')
    exit()

while(cap.isOpened()):# if not cam else True):
# while(True):
    ret, frame = cap.read()
    
    if ret == False:
        print('e')
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if prev_frame is None:
        prev_frame = frame
        continue
    
    compared = compare_frames(prev_frame, frame, draw_rectangles=True)
    prev_frame = frame

    cv2.imshow('Frame',compared)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

    
cap.release()


