import cv2, glob, random, math, numpy as np, dlib, itertools
from sklearn.svm import SVC
import os
import pickle
from sklearn.externals import joblib

# Set paths for files

faces_folder_path = "D:\\Defence Project\\emotion\\big dataset\\"

test_path = "D:\\Defence Project\\emotion\\test\\"

emotions = ["anger", "disgust", "happy", "sad", "surprise"]  # Emotion list


#Initialise predictor
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file


#Initialise SVM
# clf = SVC(kernel='poly', probability=True,
#           tol=1e-3)  # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel
clf = SVC(C=0.01, kernel='linear', decision_function_shape='ovo', probability=True)

# Function to check emotion in a file
def test_files():  # Define function to get file list, randomly shuffle it and split 80/20
    files = sorted(glob.glob(os.path.join(test_path , "*")),key=os.path.getmtime)

    print(files)

    for i in range (0,len(files)):
        print("testing file ",i+1)

        tt(cv2.imread(files[i]))

# Fetch files from folder
def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob(os.path.join(faces_folder_path + emotion, "*"))
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction

# Calculate proabilty
def tt(image):


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    clahe_image = clahe.apply(gray)
    landmarks_vectorised = get_landmarks(clahe_image)
    cdata=[]
    max_emo=[]
    cdata.append(np.array(landmarks_vectorised))
    a=clf.predict_proba(cdata)
    print("Anger proability",a[0][0] *100,"%")
    print("Disgust proability", a[0][1]*100,"%")
    # print "Fear proability", a[0][2]*100,"%"
    print("Happy proability", a[0][2]*100,"%")
    print("Sad proability", a[0][3]*100,"%")
    print("Surprised proability",a[0][4]*100,"%")


# Define facial landmarks
def get_landmarks(image):
    detections = detector(image, 1)
    #For all detected face instances individually
    for k,d in enumerate(detections):
        #get facial landmarks with prediction model
        shape = predictor(image, d)
        xpoint = []
        ypoint = []
        for i in range(17, 68):
            xpoint.append(float(shape.part(i).x))
            ypoint.append(float(shape.part(i).y))

        #center points of both axis
        xcenter = np.mean(xpoint)
        ycenter = np.mean(ypoint)
        #Calculate distance between particular points and center point
        xdistcent = [(x-xcenter) for x in xpoint]
        ydistcent = [(y-ycenter) for y in ypoint]

        #prevent divided by 0 value
        if xpoint[11] == xpoint[14]:
            angle_nose = 0
        else:
            #point 14 is the tip of the nose, point 11 is the top of the nose brigde
            angle_nose = int(math.atan((ypoint[11]-ypoint[14])/(xpoint[11]-xpoint[14]))*180/math.pi)

        #Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
        if angle_nose < 0:
            angle_nose += 90
        else:
            angle_nose -= 90

        landmarks = []
        for cx,cy,x,y in zip(xdistcent, ydistcent, xpoint, ypoint):
            #Add the coordinates relative to the centre of gravity
            landmarks.append(cx)
            landmarks.append(cy)

            #Get the euclidean distance between each point and the centre point (the vector length)
            meanar = np.asarray((ycenter,xcenter))
            centpar = np.asarray((y,x))
            dist = np.linalg.norm(centpar-meanar)

            #Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
            if x == xcenter:
                angle_relative = 0
            else:
                angle_relative = (math.atan(float(y-ycenter)/(x-xcenter))*180/math.pi) - angle_nose
                #print(anglerelative)
            landmarks.append(dist)
            landmarks.append(angle_relative)

    if len(detections) < 1:
        #In case no case selected, print "error" values
        landmarks = "error"
    return landmarks

# Split traning/testing data
def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(emotion)
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            print(item)
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                pass
            else:
                training_data.append(landmarks_vectorised)  # append image array to training data list
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                pass
            else:
                prediction_data.append(landmarks_vectorised)
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


accur_lin = []

for i in range(0, 1):
    print("Making sets %s" % i)  # Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print("after make sets")

    npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" % i)  # train SVM
    clf.fit(npar_train, training_labels)
    joblib.dump(clf, 'big_dataset_linear.pkl')

    print("getting accuracies %s" % i)  # Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    # print "predict ",clf.predict_proba(npar_pred)
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin)  # Store accuracy in a list



print("Mean value lin svm: %.3f" % np.mean(accur_lin))  # Get mean accuracy of the 10 runs

print("beign testing")

test_files()


