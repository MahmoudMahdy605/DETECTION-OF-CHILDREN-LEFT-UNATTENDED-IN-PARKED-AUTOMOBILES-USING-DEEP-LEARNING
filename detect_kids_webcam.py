from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# load model
model = load_model(
    'D:\FYP\Dataset\kids_detection_model\kids_detection.model')

# open webcam
webcam = cv2.VideoCapture(0)

classes = ['kids', 'Adults']

# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    kids = 0
    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for children detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply children detection on face
        # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        conf = model.predict(face_crop)[0]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        if(label == "kids"):
            label = "Adult"
        else:
            label = "kid"
            kids = kids + 1

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    if(kids == len(face) and kids > 0):
        cv2.putText(frame, "Warning kids Alone !", (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 250), 5)

    # display output
    cv2.imshow("kids detection", frame)

    print("kids: ", kids)
    if(kids == len(face)):
        print("all kids")

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
