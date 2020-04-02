# Web streaming example
# Source code from the official PiCamera package
# http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming

import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server

from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

PAGE="""\
<html>
<head>
<title>Raspberry Pi - Surveillance Camera</title>
</head>
<body>
<center><h1>Raspberry Pi - Surveillance Camera</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
</body>
</html>
"""
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


class StreamingOutput(object):
    def __init__(self, FaceApp):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.app = FaceApp

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            # Truncate returns current stream position
            
            self.buffer.truncate() 
            with self.condition:
                # get value at current frame position
                self.frame = self.buffer.getvalue()
                self.frame = detectFace(self.frame)
                self.condition.notify_all()

            # Offset the frame to start of the stream
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

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

    with picamera.PiCamera(resolution='640x480', framerate=24) as camera:
        output = StreamingOutput(app)
        #Uncomment the next line to change your Pi's Camera rotation (in degrees)
        #camera.rotation = 90
        camera.start_recording(output, format='mjpeg')
        try:
            address = ('', 8000)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        finally:
            camera.stop_recording()

