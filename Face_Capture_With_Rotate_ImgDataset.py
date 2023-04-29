# -------------------- THIS IS USED TO CAPTURE STORE THE PHOTOS TO TRAIN THE FACE RECOGNITION SYSTEMS ------------------
# ------------SPECIAL ADDITIONS ARE MADE TO SAVE IMAGES ONLY WITH CORRECT ILLUMINATION AND CORRECT TILTED HEADS---------
# -------------------------------------- BY The Strategists AKA TY-:D ----------------------------------------------

import cv2
import numpy as np
import NameFind
import os

WHITE = [255, 255, 255]

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

ID = NameFind.AddName()
Count = 0

# Capture 100 Images for data extraction from a preloaded Dataset to create your own dataset

data_set_path = "/Users/tanmayverma/Desktop/Celebrity Faces Dataset/Will Smith"  # Replace with your data set path

for filename in os.listdir(data_set_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Process only image files
        filepath = os.path.join(data_set_path, filename)
        img = cv2.imread(filepath)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.average(gray) > 101:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                FaceImage = gray[y - int(h / 2): y + int(h * 1.5), x - int(x / 2): x + int(w * 1.5)]
                Img = (NameFind.DetectEyes(FaceImage))
                cv2.putText(gray, "FACE DETECTED", (x+(w//2), y-5), cv2.FONT_HERSHEY_DUPLEX, .4, WHITE)
                if Img is not None:
                    frame = Img
                else:
                    frame = gray[y: y+h, x: x+w]
                cv2.imwrite("celebritydataSet/User." + str(ID) + "." + str(Count) + ".jpg", frame)
                cv2.waitKey(300)
                cv2.imshow("CAPTURED PHOTO", frame)
                Count = Count + 1
        cv2.imshow('Face Recognition System Capture Faces', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print ('FACE CAPTURE FOR THE SUBJECT IS COMPLETE')
cv2.destroyAllWindows()
