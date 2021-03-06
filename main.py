import cv2
import time
import face_recognition
import os
from tqdm import tqdm
import subprocess
import random


class FaceTools:
    def __init__(self):
        self.face_encodings = []
        self.face_names = []
        self.current_face_names = []
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
        self.fullscreen = False
        if self.fullscreen:
            cv2.namedWindow("Live Facial rec", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Live Facial rec", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def load_faces(self):
        print("Loading Faces. This will take a long time")
        for folder in tqdm(os.listdir("Faces")):
            for image in os.listdir("Faces/" + folder):
                self.face_encodings += face_recognition.face_encodings(face_recognition.load_image_file(
                    f"Faces/{folder}/{image}"))
                self.face_names += folder

    def save_face(self, image):
        filename = str(time.time()).replace(".", "") + ".png"
        face_id = random.randint(1000, 9999)
        cv2.imwrite(filename=filename, img=image)
        if not os.path.exists(f"Faces/{face_id}"):
            os.makedirs(f"Faces/{face_id}")
        os.rename(filename, f"Faces/{face_id}/{filename}")
        self.face_encodings += face_recognition.face_encodings(image)[0]

    def find_face_ids(self, image):
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.face_names[first_match_index]
            self.current_face_names += name
            print(name)
            print(self.current_face_names)

    def check_faces(self):
        ret, self.img = face.cap.read()
        faces = face.face_cascade.detectMultiScale(self.img, scaleFactor=1.3, minNeighbors=4)
        if len(faces) != 0:
            return True
        else:
            return False


face = FaceTools()
face.load_faces()

while True:
    if face.check_faces():
        print("HI")
        face.find_face_ids(face.img)
        print(face.face_names)
        for face in face.face_names:
            if face == "Unknown":
                face.save_face(face.img)
                break
    cv2.imshow("Live Facial rec", face.img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
