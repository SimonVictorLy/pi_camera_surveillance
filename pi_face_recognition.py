# USAGE
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
# import picamera

class FaceRecognizer:
    def __init__(self):

        # load the faces/embeddings along with OpenCV's Haar cascade for face detection
        print("[INFO] loading encodings + face detector...")
        self.data = pickle.loads(open(args["encodings"], "rb").read())
        self.detector = cv2.CascadeClassifier(args["cascade"])

    def detectFace(self,image):
        # detect face from image
        image = imutils.resize(image, width=500)
        
        # Color conversions
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find faces in grayscale
        self.rects = self.detector.detectMultiScale(self.gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # Reorder OpenCV boxes to rect requirements
        self.boxes = [(y, x + w, y + h, x) for (x, y, w, h) in self.rects]

        # compute the facial embeddings for each face bounding box
        self.encodings = face_recognition.face_encodings(self.rgb, self.boxes)
        self.names = []

        # loop over the facial embeddings
        for encoding in self.encodings:

            # attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(self.data["encodings"], encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # Cast votes
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # Most votes gets the name
                name = max(counts, key=counts.get)
                self.names.append(name)

            # Put names where the boxes are
            for ((top, right, bottom, left), name) in zip(self.boxes, self.names):

                # draw the predicted face name on the image
                cv2.rectangle(image, (left, top), (right, bottom),(0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.75, (0, 255, 0), 2)
        return image

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cascade", required=True,
            help = "path to where the face cascade resides")
    ap.add_argument("-e", "--encodings", required=True,
            help="path to serialized db of facial encodings")
    args = vars(ap.parse_args())
    
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    
    app = FaceRecognizer()
    test = cv2.imread("dataset/adrian/00000.png")
    test = app.detectFace(test)
    cv2.imshow("Test Image", test)
    cv2.waitKey(0)
    # vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    # time.sleep(2.0)


# Clean up
# cv2.destroyAllWindows()
# vs.stop()
