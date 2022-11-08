import os
import string
import random
import cv2
DIRECTORY = r"C:\Users\hoaso\OneDrive\Creative Cloud Files\Desktop\thigiacmaytinh\dataset\with_mask"

number = 0

for file in os.listdir(DIRECTORY):
    image = cv2.imread(os.path.join(DIRECTORY, file))
    print (image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    print("[INFO] Found {0} Faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

    number += 1
    status = cv2.imwrite('faces_detected' + str(number) + '.jpg', image)
    print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

    print(file)
