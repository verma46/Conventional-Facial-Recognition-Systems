# -------------------------- FACE DETECTION USING HAAR CASCADES ---------------------------
# ---------------------------- BY The Strategists AKA TY-:D -----------------------------------------

import cv2                  # Importing the opencv
import NameFind

# import the Haar cascades for face and eye ditection

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 3)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
      
    return frame

#Get a reference to webcam 
video_capture = cv2.VideoCapture(0)

while True:
    
  #Grab a single frame of video
  _, frame = video_capture.read()
  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
  canvas = detect(gray, frame)
    
  #Display the resulting image
  cv2.imshow("The Strategists", canvas)
    
  #Hit 'q' on the keyboard to quit!
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
    
#Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
#cv2.destroyAllWindows()
