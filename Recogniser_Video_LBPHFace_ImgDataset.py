# Import the required libraries
import cv2
import numpy as np
import NameFind
import time

# Import the Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

# Create LBPH Face Recognizer object and load the training data from the trainer to recognize the faces
recognizer = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 15)
recognizer.read("Recogniser/trainingDataLBPH.xml")

# Start the video feed
cap = cv2.VideoCapture(0)

buff_name='Initial'
Celeb_dict = {}
count=0

while True:
    # Read the camera object
    ret, img = cap.read()
    # Convert the Camera to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces and store the positions
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # The Face is isolated and cropped
        gray_face = gray[y: y+h, x: x+w]
        # Detect the eyes within the face
        eyes = eye_cascade.detectMultiScale(gray_face)
        for (ex, ey, ew, eh) in eyes:
            # Determine the ID of the photo
            
            ID, conf = recognizer.predict(gray_face)
            # Get the name of the matched image
            NAME = NameFind.ID2Name(ID, conf)
            # Display the name of the person with whom the image matches
            # NameFind.DispID(x, y, w, h, NAME, gray)

            
            similar_img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(similar_img, NAME, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Load the image that matched from the database and display it next to the name
            print("Id is :-",str(ID))
            print("Name is :-",NAME)
            print("Confidence : ",conf)
            print("buff name :-",buff_name)

            print("Reading in image file...") 
            img2 = cv2.imread("celebritydataSet/User." + str(ID) +".0" + ".jpg")
            if img2 is None:
                print("Error: Unable to open image")
                continue
            else:
                print(f"Image shape: {img2.shape}")
            
            print("Performing some operation on image...")

            matched_img = cv2.imread("celebritydataSet/User." + str(ID) +".0" + ".jpg")
            print(f"Matched image shape: {matched_img.shape}")

            
            # cv2.imshow(NAME, matched_img)
            if matched_img is not None and matched_img.shape[0] > 0 and matched_img.shape[1] > 0:
                cv2.imshow(NAME, matched_img)
            else:
                print("Error: Image has invalid dimensions")

            delimiter2= " Distance"
            if NAME.split(delimiter2)[0] in Celeb_dict:
               count = Celeb_dict[NAME.split(delimiter2)[0]]
               Celeb_dict[NAME.split(delimiter2)[0]] = count + 1  
            else:
            # Handle the case when the key does not exist
               if (count==0 or count>0):
                   Celeb_dict[NAME.split(delimiter2)[0]] = count + 1             
               else:
                   count=0
                   Celeb_dict[NAME.split(delimiter2)[0]] = count

            

            if  NAME.split(delimiter2)[0] == buff_name:
                continue
            else:
                cv2.destroyWindow(NAME)
                print("Image has changed")
                # cv2.imshow(NAME, matched_img)
                buff_name = NAME.split(delimiter2)[0]
                # buff_name = NAME
                print("New Name is",buff_name)
                break
                  
    # Show the video
    cv2.imshow('LBPH Face Recognition System', img)
    
    for key, value in Celeb_dict.items():
        print(key, ":", value)

    if Celeb_dict:
       max_key = max(Celeb_dict, key=Celeb_dict.get)
       max_value = Celeb_dict[max_key]
       print(f"Your face was most similar to '{max_key}' and it matched {max_value} times in our dataset")
    else:
       print("The dictionary is empty!")

    # Quit if the key is Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()