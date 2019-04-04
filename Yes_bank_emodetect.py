import cv2, glob, random, math, numpy as np, dlib, itertools

from sklearn.svm import SVC
import os
import pickle
from sklearn.externals import joblib

# Set paths for files


test_path = "D:\\Defence Project\\emotion\\test\\"

# Initialise predictor
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Or set this to whatever you named the downloaded file


# Function to check emotion in a file
def test_files():
    # Define function to get file list, randomly shuffle it and split 80/20
    files = sorted(glob.glob(os.path.join(test_path, "*")), key=os.path.getmtime)

    print(files)

    for i in range(0, len(files)):
        print ("testing file ", files[i])

        tt(cv2.imread(files[i]))

# Fetch files from folder

def tt(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    clahe_image = clahe.apply(gray)
    landmarks_vectorised = get_landmarks(clahe_image)
    cdata = []
    max_emo = []
    cdata.append(np.array(landmarks_vectorised))
    a = clf.predict_proba(cdata)
    print("Anger probability", a[0][0] * 100, "%")
    print("Disgust probability", a[0][1] * 100, "%")
    # print "Fear probability", a[0][2]*100, "%"
    print("Happy probability", a[0][2] * 100, "%")
    print("Sad probability", a[0][3] * 100, "%")
    print("Surprised probability", a[0][4] * 100, "%")


# Define facial landmarks
def get_landmarks(image):
    detections = detector(image, 1)
    # For all detected face instances individually
    landmarks = []
    for k, d in enumerate(detections):
        # get facial landmarks with prediction model
        shape = predictor(image, d)
        xpoint = []
        ypoint = []
        for i in range(17, 68):
            xpoint.append(float(shape.part(i).x))
            ypoint.append(float(shape.part(i).y))

        # center points of both axis
        xcenter = np.mean(xpoint)
        ycenter = np.mean(ypoint)
        # Calculate distance between particular points and center point
        xdistcent = [(x - xcenter) for x in xpoint]
        ydistcent = [(y - ycenter) for y in ypoint]

        # prevent divided by 0 value
        if xpoint[11] == xpoint[14]:
            angle_nose = 0
        else:
            # point 14 is the tip of the nose, point 11 is the top of the nose brigde
            angle_nose = int(math.atan((ypoint[11] - ypoint[14]) / (xpoint[11] - xpoint[14])) * 180 / math.pi)

        # Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
        if angle_nose < 0:
            angle_nose += 90
        else:
            angle_nose -= 90

        landmarks = []
        for cx, cy, x, y in zip(xdistcent, ydistcent, xpoint, ypoint):
            # Add the coordinates relative to the centre of gravity
            landmarks.append(cx)
            landmarks.append(cy)

            # Get the euclidean distance between each point and the centre point (the vector length)
            meanar = np.asarray((ycenter, xcenter))
            centpar = np.asarray((y, x))
            dist = np.linalg.norm(centpar - meanar)

            # Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
            if x == xcenter:
                angle_relative = 0
            else:
                angle_relative = (math.atan(float(y - ycenter) / (x - xcenter)) * 180 / math.pi) - angle_nose
                # print(anglerelative)
            landmarks.append(dist)
            landmarks.append(angle_relative)

    if len(detections) < 1:
        # In case no case selected, print "error" values
        landmarks = "error"
    return landmarks


clf = joblib.load('big_dataset_linear.pkl')

print("begin testing")

test_files()
