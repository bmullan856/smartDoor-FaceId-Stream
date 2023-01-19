#Modified by smartbuilds.io
#Date: 27.09.20
#Desc: This scrtipt script..

import cv2 as cv
from imutils.video.pivideostream import PiVideoStream
import imutils
import time
from datetime import datetime
import numpy as np

# facial recognition imports
import face_recognition
import pickle

#Imports need to Controle electrical Lock
import RPi.GPIO as GPIO
# set up the pins
lockPin = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(lockPin, GPIO.OUT)
           

class VideoCamera(object):
    def __init__(self, flip = False, file_type  = ".jpg", photo_string= "stream_photo"):
        # self.vs = PiVideoStream(resolution=(1920, 1080), framerate=30).start()
        self.vs = PiVideoStream().start()
        self.flip = flip # Flip frame vertically
        self.file_type = file_type # image type i.e. .jpg
        self.photo_string = photo_string # Name to save the photo
        time.sleep(2.0)

    def __del__(self):
        self.vs.stop()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame(self):
        #Initialize 'currentname' to trigger only when a new person is identified.
        currentname = "unknown"
        #Determine faces from encodings.pickle file model created from train_model.py
        encodingsP = "encodings.pickle"

        # load the known faces and embeddings along with OpenCV's Haar
        # cascade for face detection
        data = pickle.loads(open(encodingsP, "rb").read())
        
        #Framing the camera module
        frame = self.flip_if_needed(self.vs.read())
        
        #*******Facial reconition code*********
        
        #Gets the X and Y location of any face on the stream         
        boxes = face_recognition.face_locations(frame)
        
        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(frame, boxes)
        
        #Stores all the name of people that ahve been reconised         
        names = []
        
        # loop over the facial embeddings
        for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings

            #compares data to see if the live feed is match
            matches = face_recognition.compare_faces(data["encodings"],encoding)
            name = "Unknown" #if face is not recognized, then print Unknown

            # check to see if we have found a match
            if True in matches:
            # Unlock door                   
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
                print("face reconised")
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                
                print('before')
                #unlocking door
                GPIO.output(lockPin, False)
                time.sleep(2)
                GPIO.output(lockPin, True)
                time.sleep(2)
                print('after')
            
            #what to do if there is a match           
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

            #If someone in your dataset is identified, print their name on the screen
            if currentname != name:
                currentname = name
                print(currentname)  
                
                
            # update the list of names
            names.append(name)
            
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image - color is in BGR
            cv.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv.putText(frame, name, (left, y), cv.FONT_HERSHEY_SIMPLEX,
                .8, (0, 255, 255), 2)
        
        ret, jpeg = cv.imencode(self.file_type, frame)
        self.previous_frame = jpeg
        
        return jpeg.tobytes()

    # Take a photo, called by camera button
    def take_picture(self):
        frame = self.flip_if_needed(self.vs.read())
        ret, image = cv.imencode(self.file_type, frame)
        today_date = datetime.now().strftime("%m%d%Y-%H%M%S") # get current time
        cv.imwrite(str(self.photo_string + "_" + today_date + self.file_type), frame)
