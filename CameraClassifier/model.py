from sklearn.svm import LinearSVC
import numpy as np
import PIL.Image
import cv2 as cv


class Model:
    def __init__(self):
        self.model = LinearSVC()

    def train_model(self, counter):
        img_list = []
        class_list = []

        # Check images in class 1
        for i in range(1, counter[0]):
            path = f"1/frame{i}.jpg"
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Image not found: {path}")  # Debug print
                continue  # Skip if image is not found

            img = cv.resize(img, (150, 112))  # Resize image to match expected size
            img = img.reshape(16800)  # Flatten the image

            img_list.append(img)
            class_list.append(1)

        # Check images in class 2
        for i in range(1, counter[1]):
            path = f"2/frame{i}.jpg"
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Image not found: {path}")  # Debug print
                continue  # Skip if image is not found

            img = cv.resize(img, (150, 112))  # Resize image to match expected size
            img = img.reshape(16800)  # Flatten the image

            img_list.append(img)
            class_list.append(2)

        # Convert lists to numpy arrays
        img_list = np.array(img_list)
        class_list = np.array(class_list)

        print(f"Images loaded: {img_list.shape[0]}")  # Debug print to check how many images were loaded

        # If no images were loaded, raise an error
        if img_list.shape[0] == 0:
            raise ValueError("No images were loaded. Check the image paths and try again.")

        self.model.fit(img_list, class_list)
        print("Model Successfully Trained")

    def predict(self, frame):
        """
        Make a prediction based on the input frame.

        Args:
            frame: A single flattened image or a 2D/3D image frame

        Returns:
            int: Class prediction (1 or 2)
        """
        # Handle tuple input from camera.get_frame()
        if isinstance(frame, tuple) and len(frame) == 2:
            ret, frame = frame
            if not ret:
                return 0  # Return 0 if frame couldn't be captured

        # Convert to numpy array if it's not already
        if not isinstance(frame, np.ndarray):
            try:
                frame = np.array(frame)
            except Exception as e:
                print(f"Error converting frame to numpy array: {e}")
                return 0

        # Handle different input dimensions
        if len(frame.shape) == 1:
            # Already flattened
            flat_frame = frame
        else:
            # Convert to grayscale if it's a color image
            if len(frame.shape) == 3 and frame.shape[2] >= 3:
                frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

            # Resize and flatten
            try:
                resized = cv.resize(frame, (150, 112))
                flat_frame = resized.reshape(16800)
            except Exception as e:
                print(f"Error resizing/reshaping frame: {e}")
                return 0

        # Make prediction
        try:
            prediction = self.model.predict([flat_frame])
            return prediction[0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            return 0