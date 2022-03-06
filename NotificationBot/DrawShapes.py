import numpy as np
import cv2
def DrawCircle(radius, x, y, screenNP, screenX, screenY):
    maxX = screenX - 1
    maxY = screenY - 1
    xMultipliers = [3**0.5 / 2 , 1/2]
    yMultipliers = [1/2, 3**0.5 / 2]
    for xm, ym in xMultipliers, yMultipliers:
        startX = 0 if x - round(radius * xm) < 0 else x - round(radius * xm)
        startY = 0 if y - round(radius * ym) < 0 else y - round(radius * ym)
        endX = maxX if x + round(radius * xm) > maxX else x + round(radius * xm)
        endY = maxY if y + round(radius * ym) > maxY else y + round(radius * ym)
        screenNP[startY:endY, startX:endX] = 255
    xMultipliers = [-1, -(3**0.5)/2, -1/2, 0, 1/2, (3**0.5)/2, 1, (3**0.5)/2, 1/2, 0, -1/2, -(3**0.5)/2]
    yMultipliers = [0, -1/2, -(3**0.5)/2, -1, -(3**0.5)/2, -1/2, 0, 1/2, (3**0.5)/2, 1, (3**0.5)/2, 1/2]
    xChanges = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    yChanges = [-1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
    index = 0
    while index < 6:
        startX = x + round(radius * xMultipliers[index])
        startY = y + round(radius * yMultipliers[index])
        endX = x + round(radius * xMultipliers[(index + 1) % 12])
        endY = y + round(radius * yMultipliers[(index + 1) % 12])
        currentX = startX
        currentY = startY
        while currentY != endY and currentX != endX:
            distance = (((currentX - x) ** 2) + ((currentY - y) ** 2)) ** 0.5
            while distance > radius:
                if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                    currentX += xChanges[index]
                else:
                    currentY += yChanges[index]
                distance = (((currentX - x) ** 2) + ((currentY - y) ** 2)) ** 0.5
            if currentX > maxX or currentX < 0 or currentY > maxY or currentY < 0:
                if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                    currentY += yChanges[index]
                else:
                    currentX += xChanges[index]
                continue
            if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                screenNP[currentY, currentX:endX] = 255
                currentY += yChanges[index]
            else:
                screenNP[currentY:endY, currentX] = 255
                currentX += xChanges[index]
        index += 1
    index = 6
    while index < 12:
        startX = x + round(radius * xMultipliers[index])
        startY = y + round(radius * yMultipliers[index])
        endX = x + round(radius * xMultipliers[(index + 1) % 12])
        endY = y + round(radius * yMultipliers[(index + 1) % 12])
        currentX = startX
        currentY = startY
        while currentY != endY and currentX != endX:
            distance = (((currentX - x)**2) + ((currentY - y)**2))**0.5
            while distance > radius:
                if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                    currentX += xChanges[index]
                else:
                    currentY += yChanges[index]
                distance = (((currentX - x) ** 2) + ((currentY - y) ** 2)) ** 0.5
            if currentX > maxX or currentX < 0 or currentY > maxY or currentY < 0:
                if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                    currentY += yChanges[index]
                else:
                    currentX += xChanges[index]
                continue
            if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                screenNP[currentY, endX:currentX] = 255
                currentY += yChanges[index]
            else:
                screenNP[endY:currentY, currentX] = 255
                currentX += xChanges[index]
        index += 1
    return screenNP
def HeatCircle(radius, heatAdd, x, y, screenNP, screenX, screenY):
    heatPoints = []
    maxX = screenX - 1
    maxY = screenY - 1
    xMultipliers = [3**0.5 / 2 , 1/2]
    yMultipliers = [1/2, 3**0.5 / 2]
    for xm, ym in xMultipliers, yMultipliers:
        startX = 0 if x - round(radius * xm) < 0 else x - round(radius * xm)
        startY = 0 if y - round(radius * ym) < 0 else y - round(radius * ym)
        endX = maxX if x + round(radius * xm) > maxX else x + round(radius * xm)
        endY = maxY if y + round(radius * ym) > maxY else y + round(radius * ym)
        screenNP[startY:endY, startX:endX] += heatAdd
    xMultipliers = [-1, -(3**0.5)/2, -1/2, 0, 1/2, (3**0.5)/2, 1, (3**0.5)/2, 1/2, 0, -1/2, -(3**0.5)/2]
    yMultipliers = [0, -1/2, -(3**0.5)/2, -1, -(3**0.5)/2, -1/2, 0, 1/2, (3**0.5)/2, 1, (3**0.5)/2, 1/2]
    xChanges = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    yChanges = [-1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
    index = 0
    while index < 6:
        startX = x + round(radius * xMultipliers[index])
        startY = y + round(radius * yMultipliers[index])
        endX = x + round(radius * xMultipliers[(index + 1) % 12])
        endY = y + round(radius * yMultipliers[(index + 1) % 12])
        currentX = startX
        currentY = startY
        while currentY != endY and currentX != endX:
            distance = (((currentX - x) ** 2) + ((currentY - y) ** 2)) ** 0.5
            while distance > radius:
                if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                    currentX += xChanges[index]
                else:
                    currentY += yChanges[index]
                distance = (((currentX - x) ** 2) + ((currentY - y) ** 2)) ** 0.5
            if currentX > maxX or currentX < 0 or currentY > maxY or currentY < 0:
                if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                    currentY += yChanges[index]
                else:
                    currentX += xChanges[index]
                continue
            if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                screenNP[currentY, currentX:endX] += heatAdd
                currentY += yChanges[index]
            else:
                screenNP[currentY:endY, currentX] += heatAdd
                currentX += xChanges[index]
        index += 1
    index = 6
    while index < 12:
        startX = x + round(radius * xMultipliers[index])
        startY = y + round(radius * yMultipliers[index])
        endX = x + round(radius * xMultipliers[(index + 1) % 12])
        endY = y + round(radius * yMultipliers[(index + 1) % 12])
        currentX = startX
        currentY = startY
        while currentY != endY and currentX != endX:
            distance = (((currentX - x)**2) + ((currentY - y)**2))**0.5
            while distance > radius:
                if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                    currentX += xChanges[index]
                else:
                    currentY += yChanges[index]
                distance = (((currentX - x) ** 2) + ((currentY - y) ** 2)) ** 0.5
            if currentX > maxX or currentX < 0 or currentY > maxY or currentY < 0:
                if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                    currentY += yChanges[index]
                else:
                    currentX += xChanges[index]
                continue
            if index != 3 and index != 4 and index != 5 and index != 9 and index != 10 and index != 11:
                screenNP[currentY, endX:currentX] += heatAdd
                currentY += yChanges[index]
            else:
                screenNP[endY:currentY, currentX] += heatAdd
                currentX += xChanges[index]
        index += 1
    return screenNP
#screenNP = np.zeros([1080, 1920, 3], np.uint8)
#screenNP[:] = 255
#radius = 50
#screenNP = DrawCircle(radius, 15, 45, screenNP, 1920, 1080)
#cv2.imshow("Hunter x Hunter New Chapter", screenNP)
#cv2.waitKey(0) & 0xff == ord("q")