# USAGE
# python real_time_object_detection.py

# import the necessary packages
from math import sqrt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from PIL import Image
from numpy import asarray
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import joblib




# construct the argument parse and parse the arguments
'''ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())'''

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


face_net = load_model('facenet_keras.h5')
face_pred_model = joblib.load('final_model.sav')
in_encoder = Normalizer(norm='l2')


def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = face_net.predict(samples)
    return yhat[0]



def predict_face(face_location,filename,required_size=(160, 160)):
    x1,y1,x2,y2 = face_location
    pixels = asarray(filename)
    image2 = cv2.cvtColor(filename, cv2.COLOR_RGB2BGR)
    face_pixels = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face_pixels)
    image = image.resize(required_size)
    face_array = asarray(image)
    newTrainX = list()
    embedding = get_embedding(face_pred_model, face_array)
    newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    trainX = in_encoder.transform(newTrainX)
    y_class = face_pred_model.predict(trainX)
    # print(y_class)
    y_prob = face_pred_model.predict_proba(trainX)
    # print(y_prob)
    class_index = y_class[0]
    print(class_index)
    class_probability = y_prob[0, class_index] * 100

    if class_probability < 96.00:
            name = "Unknown"
            #un += 1
    else:
        if class_index == 0:
            name = "shashank"
        elif class_index == 1:
            name = "prathap"

    print(name)        





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
        if confidence > 0.4:
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
        # for faster inference we'll make batch predictions on all
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


prototxtPath = os.path.join(os.getcwd(), "face_detector/deploy.prototxt")
weightsPath = os.path.join(
    os.getcwd(), "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    coordinates = dict()
    pos_dict = dict()
    F = 615

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.2:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                break
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (perstartX, perstartY, perendX, perendY) = box.astype("int")

            # draw the prediction on the frame
            

            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                print(withoutMask)
        # determine the class label and color we'll use to draw
        # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		        
                if withoutMask > mask:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    predict_face(box,image_rgb)

				#color = (0,255,0) if mask > withoutMask else (0,0,255)

                cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # social distance another code---------------------------------------
            coordinates[i] = (perstartX, perstartY, perendX, perendY)

        # Mid point of bounding box
            x_mid = round((perstartX+perendX)/2, 4)
            y_mid = round((perstartY+perendY)/2, 4)

            height = round(perendY-perstartY, 4)

        # Distance from camera based on triangle similarity
            distance = (165 * F)/height
            print("Distance(cm):{dist}\n".format(dist=distance))

        # Mid-point of bounding boxes (in cm) based on triangle similarity technique
            x_mid_cm = (x_mid * distance) / F
            y_mid_cm = (y_mid * distance) / F
            pos_dict[i] = (x_mid_cm, y_mid_cm, distance)

        # Distance between every object detected in a frame
        close_objects = set()
        for i in pos_dict.keys():
            for j in pos_dict.keys():
                if i < j:
                    dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0], 2) + pow(
                        pos_dict[i][1]-pos_dict[j][1], 2) + pow(pos_dict[i][2]-pos_dict[j][2], 2))

                    # Check if distance less than 2 metres or 200 centimetres
                    if dist < 200:
                        close_objects.add(i)
                        close_objects.add(j)

        for i in pos_dict.keys():
            if i in close_objects:
                COLOR = (0, 0, 255)
                """
			if not ALARM_ON:
				ALARM_ON = True
				if args["alarm"] != "":
					t = Thread(target=sound_alarm, args=(args["alarm"],))
					t.deamon = True
					t.start()
			"""
            else:
                COLOR = (0, 255, 0)
                # ALARM_ON = False
            (perstartX, perstartY, perendX, perendY) = coordinates[i]

            cv2.rectangle(frame, (perstartX, perstartY), (perendX, perendY), COLOR, 2)
            #y = startY - 15 if startY - 15 > 15 else startY + 15
            # Convert cms to feet
            #cv2.putText(frame, 'Depth: {i} ft'.format(i=round(pos_dict[i][2]/30.48, 4)), (startX, y),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
