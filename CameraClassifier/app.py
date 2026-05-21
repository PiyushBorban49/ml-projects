import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
import numpy as np
import model
import camera


class App:
    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):
        self.window = window
        self.window.title(window_title)

        self.counter = [1, 1]

        self.model = model.Model()

        self.auto_predict = False

        self.camera = camera.Camera()

        self.init_gui()

        self.delay = 15
        self.update()

        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = simpledialog.askstring("Classname one", "Enter the name of the first class: ",
                                                    parent=self.window)
        self.classname_two = simpledialog.askstring("Classname two", "Enter the name of the second class: ",
                                                    parent=self.window)

        self.btn_classname_one = tk.Button(self.window, text=self.classname_one, width=50,
                                           command=lambda: self.save_for_class(1))
        self.btn_classname_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_classname_two = tk.Button(self.window, text=self.classname_two, width=50,
                                           command=lambda: self.save_for_class(2))
        self.btn_classname_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Now", width=50,
                                   command=lambda: self.model.train_model(self.counter))
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="RESET", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="Class")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not os.path.exists('1'):
            os.mkdir('1')
        if not os.path.exists('2'):
            os.mkdir('2')

        # Save the image for training
        cv.imwrite(f"{class_num}/frame{self.counter[class_num - 1]}.jpg", cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open(f"{class_num}/frame{self.counter[class_num - 1]}.jpg")
        img.thumbnail((150, 150), PIL.Image.Resampling.LANCZOS)
        img.save(f"{class_num}/frame{self.counter[class_num - 1]}.jpg")

        self.counter[class_num - 1] += 1

    def reset(self):
        for directory in ['1', '2']:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
        self.counter = [1, 1]
        self.model = model.Model()
        self.class_label.config(text="CLASS")

    def update(self):
        if self.auto_predict:
            self.predict()

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def predict(self):
        """Capture a frame and make a prediction."""
        ret, frame = self.camera.get_frame()

        if not ret:
            print("Failed to capture frame")
            return None

        try:
            # Convert to grayscale
            gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

            # Resize and flatten
            resized = cv.resize(gray_frame, (150, 112))
            flat_frame = resized.reshape(16800)

            # Make prediction
            prediction = self.model.predict(flat_frame)

            # Update label based on prediction
            if prediction == 1:
                self.class_label.config(text=self.classname_one)
                return self.classname_one
            elif prediction == 2:
                self.class_label.config(text=self.classname_two)
                return self.classname_two
            else:
                self.class_label.config(text="Unknown")
                return "Unknown"
        except Exception as e:
            print(f"Error in prediction: {e}")
            self.class_label.config(text="Error")
            return "Error"