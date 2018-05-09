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

    @staticmethod
    def save_face(image):
        filename = str(time.time()).replace(".", "") + ".png"
        face_id = random.randint(1000, 9999)
        cv2.imwrite(filename=filename, img=image)
        if not os.path.exists(f"Faces/{face_id}"):
            os.makedirs(f"Faces/{face_id}")
        os.rename(filename, f"Faces/{face_id}/{filename}")


    def find_face_ids(image):
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(rgb_small_frame, face_locations)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.face_names[first_match_index]

    def check_for_faces(self):
        ret, img = face.cap.read()
        faces = face.face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4)
        if len(faces) != 0:
            return True
        else:
            return False


face = FaceTools()
face.load_faces()

while True:

    cv2.imshow("Live Facial rec", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
