import numpy as np
import cv2
def DrawCircle(radius, x, y, screenNP):
    xMultipliers = [2 / 3**0.5, 1/2]
    yMultipliers = [1/2, 2 / 3**0.5]
    for xm, ym in xMultipliers, yMultipliers:
        startX = 0 if x - int(radius * xm) < 0 else x - int(radius * xm)
        startY = 0 if y - int(radius * ym) < 0 else y - int(radius * ym)
        endX = 1919 if x + int(radius * xm) > 1919 else x + int(radius * xm)
        endY = 1079 if y + int(radius * ym) > 1079 else y + int(radius * ym)
        screenNP[startY:endY, startX:endX] = 255
    startX = 0 if x - radius
    startY = 0 if
    endX = 1919 if
    endY = 1079 if
    return screenNP
screenNP = np.zeros([1080, 1920, 3], np.uint8)
#screenNP[:] = 255
radius = 50
screenNP = DrawCircle(50, 960, 540, screenNP)
cv2.imshow("Hunter x Hunter New Chapter", screenNP)
cv2.waitKey(0) & 0xff == ord("q")