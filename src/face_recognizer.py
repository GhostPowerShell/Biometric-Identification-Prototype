import cv2
import dlib
import face_recognition
import numpy as np
import os

from face_recognition.api import face_encoder


class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.load_known_faces()

    def load_known_faces(self):
        for filename in os.listdir('./faces'):
            image = face_recognition.load_image_file(f'./faces/{filename}')
            encoding = face_recognition.face_encodings(image)[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(filename.split('.')[0])

    def recognize_faces(self, frame):
        # rgb_frame = frame[:, :, ::-1]
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = []
        face_names = []
        for face_location in face_locations:
            top, right, bottom, left = face_location
            rect = dlib.rectangle(left, top, right, bottom)
            landmarks = self.shape_predictor(rgb_frame, rect)
            face_descriptor = np.array(face_encoder.compute_face_descriptor(rgb_frame, landmarks))
            face_encodings.append(face_descriptor)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            face_names.append(name)
        return face_locations, face_names
