import cv2
from tkinter import *
from PIL import Image, ImageTk
from tkinter import simpledialog, messagebox
from src.face_recognizer import FaceRecognizer
from src.face_database import FaceDatabase

def register_face():
    name = simpledialog.askstring("Input", "Please enter your name:")
    if name:
        ret, frame = cap.read()
        if ret:
            face_locations, _ = face_recognizer.recognize_faces(frame)
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_image = frame[top:bottom, left:right]
                face_database.save_face(face_image, name)
                messagebox.showinfo("Success", "Face registered successfully!")
                face_recognizer.load_known_faces()  # reload known faces

def update_frame():
    ret, frame = cap.read()
    if ret:
        face_locations, face_names = face_recognizer.recognize_faces(frame)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            if name == "Unknown":
                cv2.putText(frame, "User not found, Please register", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                register_button.pack()
            else:
                cv2.putText(frame, f"Welcome {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                register_button.pack_forget()

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
    lmain.after(10, update_frame)  # repeat

if __name__ == "__main__":
    face_recognizer = FaceRecognizer()
    face_database = FaceDatabase()
    cap = cv2.VideoCapture(0)

    root = Tk()
    lmain = Label(root)
    lmain.pack()

    register_button = Button(root, text="Register", command=register_face)

    update_frame()  # start the update loop to display webcam feed
    root.mainloop()

    cap.release()
