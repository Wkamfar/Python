from email.mime.text import MIMEText as MimeText
from email.mime.image import MIMEImage as MimeImage
from email.mime.application import MIMEApplication as MimeApp
from email.mime.multipart import MIMEMultipart as MimeMultipart
import time
import os
import smtplib
import schedule
from PIL import ImageGrab
import numpy as np
import cv2
def Message(subject, images, attachments, bodyText):
    msg = MimeMultipart()
    msg["Subject"] = subject
    msg.attach(MimeText(bodyText))
    if images is not None:
        for image in images:
            data = open(image, 'rb').read()
            msg.attach(MimeImage(data, name = os.path.basename(image)))
    if attachments is not None:
        for attachment in attachments:
            with open(attachment, 'rb') as f:
                file = MimeApp(f.read(), name = os.path.basename(attachment))
                msg.attach(file)
                #file["C"]
    return msg
def Mail():
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.login('williampy3auto@gmail.com', 'Python0000')
    totalMessages = 100
    recipients = ["wkamfar@gmail.com"]
    images = ["C:\\Users\\wkamf\\Python\\EmailAutomation\\RespawnLogo.png", "C:\\Users\\wkamf\\Python\\EmailAutomation\\EldenRingPicture.png"]
    attachments = ["C:\\Users\\wkamf\\Python\\EmailAutomation\\main.py"]
    for i in range(0, totalMessages):
        msg = Message("Test Email " + str(i), images, None,"I am testing if this automated Email machine is working, if you receive this, please reply back to see if this is working. Thank you for your assistance.")
        smtp.sendmail(from_addr="williampy3auto@gmail.com", to_addrs=recipients, msg=msg.as_string())
        time.sleep(0.1)
    smtp.quit()
def DisplayScreen(image):
    cv2.imshow("Lelouch vi Britannia", image)
    cv2.imwrite("1.png", image)
    if cv2.waitKey(0) & 0xff == ord("q"):
        return
def GetScreen():
        screen = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
        screenNP = np.array(screen)
        # resize if you want to display
        screenHeight, screenWidth, c = screenNP.shape
        screenScale = 0.5
        screenNP = cv2.resize(screenNP, (int(screenWidth * screenScale), int(screenHeight * screenScale)))
        screenNPGray = cv2.cvtColor(screenNP, cv2.COLOR_BGR2GRAY)
        return screenNP, screenNPGray
        cv2.imshow("Lelouch Lamperouge", screenNP)
        if cv2.waitKey(0) & 0xff == ord("q"):
            return
def SeeSegments(colorImage, grayImage):
    grayImage = cv2.Canny(grayImage, 25, 100)
    ret, bw = cv2.threshold(grayImage, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    connections = 1
    components, output, stats, centers = cv2.connectedComponentsWithStats(bw, connections, cv2.CV_32S)
    sizes = stats[1:, -1]
    components -= 1
    minSize = 50
    tempImage = np.zeros((colorImage.shape), np.uint8)
    for i in range(0, components):
        if sizes[i] >= minSize:
            color = np.random.randint(255, size = 3)
            cv2.rectangle(tempImage, (stats[i][0], stats[i][1]), (stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]), (0, 30, 155), 2)
            tempImage[output == i + 1] = color
    DisplayScreen(tempImage)
def Main():
    schedule.every(1).minutes.do(Mail)
    schedule.every(1).hour.do(Mail)
    schedule.every(2).days.at("3:00").do(Mail)
    schedule.every().sunday.at("10:00").do(Mail)
    while True:
        schedule.run_pending()
        time.sleep(1)
time.sleep(1)
colorImage, grayImage = GetScreen()
SeeSegments(colorImage, grayImage)

