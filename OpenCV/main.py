import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui as auto
from skimage.metrics import structural_similarity
import time
from matplotlib import pyplot as plt

def TemplateMatching():
    print("hello, my dear listeners, Gentle Criminal is here")
    roomImage = cv2.imread("Room.jpg")
    roomHeight, roomWidth, c = roomImage.shape
    newDimension = (int(roomWidth * .75), int(roomHeight * .75))
    roomImage = cv2.resize(roomImage, newDimension, interpolation=cv2.INTER_AREA)
    roomImageGray = cv2.cvtColor(roomImage, cv2.COLOR_BGR2GRAY)
    clockImage = cv2.imread("Clock.jpg", 0)
    clockHeight, clockWidth = clockImage.shape
    print(clockImage.shape)
    detectedImage = cv2.matchTemplate(roomImageGray, clockImage, cv2.TM_CCOEFF_NORMED)
    # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(detectedImage)
    # topLeft = maxLoc
    # bottomRight = (topLeft[0] + clockWidth, topLeft[0] + clockHeight)
    # cv2.rectangle(roomImage, topLeft, bottomRight, 255, 2)
    threshold = 0.6
    # cv2.imshow("RoomGray", roomImageGray)
    # cv2.imshow("RoomColor", roomImage)
    # cv2.imshow("ClockGray", clockImage)
    # cv2.waitKey(0)
    location = np.where(detectedImage >= threshold)
    for point in zip(*location[::-1]):
        print(point)
        cv2.rectangle(roomImage, point, (point[0] + clockWidth, point[1] + clockHeight), (255, 0, 0), 2)
    # plt.subplot(121)
    cv2.imshow("Milkman", roomImage)
    cv2.waitKey(0)

def Automation():
    print(auto.position())
    #auto.moveRel(200, -300, duration = 1)
    #print(auto.position())

    #print(auto.position())
    #auto.moveTo(1617, 400, duration = 2)
    #auto.click(1617, 274)
    #auto.dragTo(100, 100, duration = 1)
    #auto.scroll(2000)
    #auto.typewrite("Bakugo died in the kidnapping!")
    #auto.typewrite(["a", "left", "ctrlleft", "a"])
    #auto.hotkey("ctrlleft", "a")
    #auto.hotkey("ctrlleft", "x")
def Screenshot():
    while True:
        screen = ImageGrab.grab(bbox = (0, 0, 1920, 1080))
        screenNP = np.array(screen)
        screenNP = cv2.cvtColor(screenNP, cv2.COLOR_BGR2RGB)
        shotHeight, shotWidth, c = screenNP.shape
        newDimension = (int (shotWidth * .25), int (shotHeight * .25))
        screenNP = cv2.resize(screenNP, newDimension, interpolation = cv2.INTER_AREA)
        cv2.imshow("Sir Nighteye", screenNP)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()
def WilliamResize(image, scale):
    height, width, c = image.shape
    newDimension = (int(width * scale), int(height * scale))
    image = cv2.resize(image, newDimension, interpolation=cv2.INTER_AREA)
    return image
def FindPillow():
    while True:
        screen = ImageGrab.grab(bbox = (0, 0, 1920, 1080))
        screenNP = np.array(screen)
        screenNP = cv2.cvtColor(screenNP, cv2.COLOR_BGR2RGB)
        screenNP = WilliamResize(screenNP, .25)
        #time.sleep(3)
        TemplateImage = cv2.imread("XButton.jpg")
        TemplateImage = WilliamResize(TemplateImage, 0.25)
        TemplateHeight, TemplateWidth, c = TemplateImage.shape
        print(TemplateImage.shape)
        detectedImage = cv2.matchTemplate(screenNP, TemplateImage, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        location = np.where(detectedImage >= threshold)
        for point in zip(*location[::-1]):
            print(point)
            cv2.rectangle(screenNP, point, (point[0] + TemplateWidth, point[1] + TemplateHeight), (255, 0, 0), 2)
            break
        #print(auto.position())
        cv2.imshow("Sir Nighteye", screenNP)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()
def Filter():
    while True:
        screen = ImageGrab.grab(bbox = (0, 0, 1920, 1080))
        screenNP = np.array(screen)
        screenNP = cv2.cvtColor(screenNP, cv2.COLOR_BGR2RGB)
        screenNP = WilliamResize(screenNP, .4)
        blur = cv2.blur(screenNP, (1, 1))
        edges = cv2.Canny(blur, 25, 100)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(screenNP, -1, kernel)
        cv2.imshow("Hatsume Mei", sharpen)
        #cv2.imshow("Izuku Midoriya", blur)
        #cv2.imshow("Mirio Totoga", edges)
        cv2.imshow("Sir Nighteye", screenNP)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()
def OrbFeatureMatching(threshold):
    image1Path = "Images\\Outliers_Book\\1.jpg"
    image2Path = "Images\\Outliers_Book\\2.jpg"
    image1Color = cv2.imread(image1Path)
    image2Color = cv2.imread(image2Path)
    image1 = cv2.imread(image1Path, 0)
    image2 = cv2.imread(image2Path, 0)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptors1, descriptors2)
    good = []
    for i, m in enumerate(matches):
        if i < len(matches) - 1 and m.distance < threshold * matches[i + 1].distance:
            good.append(m)
    finalImage = cv2.drawMatches(image1Color, keypoints1, image2Color, keypoints2, good, None)
    finalImage = cv2.resize(finalImage, (1000, 600))
    cv2.imshow("Nezuko Kamado", finalImage)
    if cv2.waitKey(0) & 0xff == ord("q"):
        cv2.destroyAllWindows()
        return
def OrbMatching(threshold, screenImage, templateImage):
    screenImageGray = cv2.cvtColor(screenImage, cv2.COLOR_BGR2GRAY)
    templateImageGrey = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(screenImageGray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(templateImageGrey, None)
    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptors1, descriptors2)
    good = []
    for i, m in enumerate(matches):
        if i < len(matches) - 1 and m.distance < threshold * matches[i + 1].distance:
            good.append(m)
    finalImage = cv2.drawMatches(screenImage, keypoints1, templateImage, keypoints2, good[:], None)
    finalImage = cv2.resize(finalImage, (int (1920 / 3), int (1080 / 3)))
    cv2.imshow("Nezuko Kamado", finalImage)
    cv2.waitKey(500)
    #if cv2.waitKey(1) & 0xff == ord("q"):
        #cv2.destroyAllWindows()
        #return
def RuntimeMatching(threshold) :
    templateImage = cv2.imread("Images\\Outliers_Book\\1.jpg")
    templateHeight, templateWidth, c = templateImage.shape
    #templateImage = cv2.resize(templateImage, (500, 300))
    while True:
        screen = ImageGrab.grab(bbox = (0, 0, 1920 / 2, 1080))
        screenNP = np.array(screen)
        screenNP = cv2.cvtColor(screenNP, cv2.COLOR_BGR2RGB)
        screenNP = cv2.resize(screenNP, (templateWidth, templateHeight))
        OrbMatching(threshold, screenNP, templateImage)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()
def SiftFeatureMatching(threshold):
    image1Path = "Images\\Others\\Art_of_War.jpg"
    image2Path = "Images\\Outliers_Book\\2.jpg"
    image1Color = cv2.imread(image1Path)
    image2Color = cv2.imread(image2Path)
    image1 = cv2.imread(image1Path, 0)
    image2 = cv2.imread(image2Path, 0)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k = 2)
    good = []
    for m, n in matches :
        if m.distance < threshold * n.distance :
            good.append([m])
    finalImage = cv2.drawMatchesKnn(image1Color, keypoints1, image2Color, keypoints2, good, None)
    finalImage = cv2.resize(finalImage, (1000, 600))
    cv2.imshow("Nezuko Kamado", finalImage)
    if cv2.waitKey(0) & 0xff == ord("q"):
        cv2.destroyAllWindows()
        return
def TwoImageDiff():
    screen = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    screenNP1 = np.array(screen)
    screenNP1 = WilliamResize(screenNP1, 0.5)
    screenNP1 = cv2.cvtColor(screenNP1, cv2.COLOR_BGR2GRAY)
    while True:
        screen = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
        screenNP2 = np.array(screen)
        screenNP2 = WilliamResize(screenNP2, 0.5)
        screenNP2 = cv2.cvtColor(screenNP2, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(screenNP1, screenNP2, full = True)
        diff = (diff * 255).astype("uint8")
        screenNP1 = screenNP2.copy()
        #print("ssim is: ", score)
        cv2.imshow("Fumikage Tokoyami", diff)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()
def main():
    #Automation()
    #Screenshot()
    #FindPillow()
    #Filter()
    #OrbFeatureMatching(0.7)
    #SiftFeatureMatching(0.5)
    #RuntimeMatching(0.55)
    TwoImageDiff()

if __name__ == '__main__':
    main()