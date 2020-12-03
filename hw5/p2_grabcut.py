import numpy as np
import cv2
import sys

# parse system arguments
img = cv2.imread(sys.argv[1])
coords = np.loadtxt(sys.argv[2])

# prepare recorder
big_box = None
foregrounds = []
backgrounds = []

# classify each row
for row in coords:
    xs = [row[1], row[3], row[5], row[7]]
    ys = [row[2], row[4], row[6], row[8]]
    upper_left = [int(min(xs)), int(min(ys))]
    lower_right = [int(max(xs)), int(max(ys))]
    # if first digit of row is 0 then foreground
    if row[0] == 0:
        foregrounds.append([upper_left, lower_right])
    # if first digit of row is 1 then background
    elif row[0] == 1:
        backgrounds.append([upper_left, lower_right])
    # if first digit of row is 2 then frame of the object
    elif row[0] == 2:
        big_box = [upper_left, lower_right]

cpy = img.copy()
# create mask
mask = np.zeros(img.shape[:2],np.uint8)
# foreground and background models
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
# rectangle of frame of objects
rect = (big_box[0][0],big_box[0][1],big_box[1][0],big_box[1][1])
# grab cut
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
# draw rectangle of the object frame
cv2.rectangle(cpy, (big_box[0][0], big_box[0][1]), (big_box[1][0], big_box[1][1]), (255, 0, 0), 5)
cv2.putText(cpy,"FRAME", (big_box[0][0],big_box[0][1]), cv2.FONT_HERSHEY_PLAIN , 2, (255, 0, 0), 0)
# mask all backgrounds
for rec in backgrounds:
    newmask = np.zeros(img.shape[:2],np.uint8)
    newmask[rec[0][1]:rec[1][1], rec[0][0]:rec[1][0]] = 1
    mask[newmask == 1] = 0
    # draw background rectangles
    cv2.rectangle(cpy, (rec[0][0], rec[0][1]), (rec[1][0], rec[1][1]), (0, 255, 0), 5)
    cv2.putText(cpy, "BACKGROUND", (rec[0][0], rec[0][1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 0)
# mask all foregrounds
for rec in foregrounds:
    newmask = np.zeros(img.shape[:2], np.uint8)
    newmask[rec[0][1]:rec[1][1], rec[0][0]:rec[1][0]] = 1
    mask[newmask == 1] = 1
    # draw all foreground rectangles
    cv2.rectangle(cpy, (rec[0][0], rec[0][1]), (rec[1][0], rec[1][1]), (0, 0, 255), 5)
    cv2.putText(cpy, "FOREGROUND", (rec[0][0], rec[0][1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 0)
# grab cut
mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
# apply mask to image
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:, :, np.newaxis]
# save mask and image to output
cv2.imwrite('grabcut_images/res.jpg', img)
cv2.imwrite('grabcut_images/box.jpg', cpy)
