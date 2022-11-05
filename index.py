#Thêm thư viện tkinter
from tkinter import *
from tkinter import filedialog
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from pygame import mixer
mixer.init()
sound = mixer.Sound('alarm.wav')


#Tạo một cửa sổ mới
window = Tk()

#Thêm tiêu đề cho cửa sổ
window.title('Detect Mask')

#Đặt kích thước của cửa sổ
window.geometry('500x500')


btn1 = Button(window, text='Select Image', width=20, height=2)


#Thêm nút chọn ảnh từ webcam
btn2 = Button(window, text='Webcam', width=20, height=2)

btn1.pack(side=TOP, pady=10)
btn2.pack(side=TOP, pady=10)



    

def checkMask():
    print("check mask")


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# add click for btn1
def selectImage():
    #get file path
    filename = filedialog.askopenfilename()
    print(filename)


    #wait for select image
    time.sleep(1)
    
    image = cv2.imread(filename)
    image = imutils.resize(image, width=400)
    (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)
    label = "No mask"
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (mask, withoutMask) = pred
        if mask > withoutMask:
            label = "Mask"
        else:
            label = "No Mask"
    #show popup notification
    if label == "No Mask":
        window1 = Tk()
        window1.title('Notification')
        window1.geometry('300x100')
        lbl1 = Label(window1, text=label, font=('Arial Bold', 20))
        lbl1.pack()
        window1.mainloop()
    else:
        window2 = Tk()
        window2.title('Notification')
        window2.geometry('300x100')
        lbl2 = Label(window2, text=label, font=('Arial Bold', 20))
        lbl2.pack()
        window2.mainloop()


def detectMaskFromWebcam():
    print("detect mask from webcam")
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")

    #check if webcam is available if not show popup error
    if not os.path.exists("/dev/video0"):
        window3 = Tk()
        window3.title('Notification')
        window3.geometry('300x100')
        lbl3 = Label(window3, text="Webcam not found", font=('Arial Bold', 20))
        lbl3.pack()
        window3.mainloop()


    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

#add click for btn2
btn2.config(command=detectMaskFromWebcam)
btn1.config(command=selectImage)



#Lặp vô tận để hiển thị cửa sổ
window.mainloop()