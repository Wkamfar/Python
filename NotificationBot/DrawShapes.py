import numpy as np
import cv2
def DrawCircle(radius, x, y, screenNP):
    xMultipliers = [3**0.5 / 2 , 1/2]
    yMultipliers = [1/2, 3**0.5 / 2]
    for xm, ym in xMultipliers, yMultipliers:
        startX = 0 if x - int(radius * xm) < 0 else x - int(radius * xm)
        startY = 0 if y - int(radius * ym) < 0 else y - int(radius * ym)
        endX = 1919 if x + int(radius * xm) > 1919 else x + int(radius * xm)
        endY = 1079 if y + int(radius * ym) > 1079 else y + int(radius * ym)
        #screenNP[startY:endY, startX:endX] = 255
    xMultipliers = [-1, -(3**0.5)/2, -1/2, 0, 1/2, (3**0.5)/2, 1, (3**0.5)/2, 1/2, 0, -1/2, -(3**0.5)/2]
    yMultipliers = [0, -1/2, -(3**0.5)/2, -1, -(3**0.5)/2, -1/2, 0, 1/2, (3**0.5)/2, 1, (3**0.5)/2, 1/2]
    xChanges = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    yChanges = [-1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
    index = 0
    while index < 12:
        print(index)
        startX = x + int(radius * xMultipliers[index])
        startY = y + int(radius * yMultipliers[index])
        endX = x + int(radius * xMultipliers[(index + 1) % 12])
        endY = y + int(radius * yMultipliers[(index + 1) % 12])
        currentX = startX
        currentY = startY
        while currentY != endY and currentX != endX:
            distance = (((currentX - x)**2) + ((currentY - y)**2))**0.5
            while distance > radius:
                if index != 2 and index != 3 and index != 8 and index != 9:
                    currentX += xChanges[index]
                else:
                    currentY += yChanges[index]
                distance = (((currentX - x) ** 2) + ((currentY - y) ** 2)) ** 0.5
            if index != 2 and index != 3 and index != 8 and index != 9:
                screenNP[currentY, currentX:endX] = 255
                print(currentX, currentY, endX, endY, "distance", distance)
                currentY += yChanges[index]
            else:
                screenNP[currentY:endY, currentX] = 255
                print(currentX, currentY, endX, endY, "distance", distance)
                currentX += xChanges[index]
        index += 1
    return screenNP
screenNP = np.zeros([1080, 1920, 3], np.uint8)
#screenNP[:] = 255
radius = 50
screenNP = DrawCircle(50, 960, 540, screenNP)
cv2.imshow("Hunter x Hunter New Chapter", screenNP)
cv2.waitKey(0) & 0xff == ord("q")