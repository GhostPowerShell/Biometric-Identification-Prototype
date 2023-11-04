import cv2
import os

class FaceDatabase:
    def __init__(self, folder_path='./faces/'):
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def save_face(self, face, name):
        filename = f"{self.folder_path}{name}.png"
        cv2.imwrite(filename, face)

    def is_registered(self, name):
        filename = f"{self.folder_path}{name}.png"
        return os.path.exists(filename)
