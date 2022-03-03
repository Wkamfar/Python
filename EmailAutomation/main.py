from email.mime.text import MIMEText as MimeText
from email.mime.image import MIMEImage as MimeImage
from email.mime.application import MIMEApplication as MimeApp
from email.mime.multipart import MIMEMultipart as MimeMultipart
import time
import os
import smtplib
import schedule

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
    totalMessages = 1
    recipients = ["ck3661@trevor.org"]
    images = ["C:\\Users\\wkamf\\Python\\EmailAutomation\\RespawnLogo.png", "C:\\Users\\wkamf\\Python\\EmailAutomation\\EldenRingPicture.png"]
    attachments = ["C:\\Users\\wkamf\\Python\\EmailAutomation\\main.py"]
    for i in range(0, totalMessages):
        msg = Message("Test Email " + str(i), images, None,"I am testing if this automated Email machine is working, if you receive this, please reply back to see if this is working. Thank you for your assistance.")
        smtp.sendmail(from_addr="williampy3auto@gmail.com", to_addrs=recipients, msg=msg.as_string())
        time.sleep(0.1)
    smtp.quit()
def Main():
    schedule.every(1).minutes.do(Mail)
    schedule.every(1).hour.do(Mail)
    schedule.every(2).days.at("3:00").do(Mail)
    schedule.every().sunday.at("10:00").do(Mail)
    while True:
        schedule.run_pending()
        time.sleep(1)
#Main()
Mail()

