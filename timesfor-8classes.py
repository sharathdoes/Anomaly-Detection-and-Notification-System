import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import requests
from io import BytesIO
from PIL import Image

# Define the custom I3DWrapperLayer
class I3DWrapperLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(I3DWrapperLayer, self).__init__(**kwargs)
        self.i3d_base_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
        self.i3d_base = hub.KerasLayer(self.i3d_base_url, trainable=False)
        
    def call(self, inputs):
        return self.i3d_base(inputs)

# Define the custom TimeSformerWrapperLayer
class TimeSformerWrapperLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TimeSformerWrapperLayer, self).__init__(**kwargs)
        # Initialize your TimeSformer specific attributes here
        
    def call(self, inputs):
        # Implement the forward pass
        pass

# Use custom object scope to load the model with the custom layers
with tf.keras.utils.custom_object_scope({
    'I3DWrapperLayer': I3DWrapperLayer,
    'TimeSformerWrapperLayer': TimeSformerWrapperLayer
}):
    i3d_model = tf.keras.models.load_model('timesformer-8classes.h5')

# Load the labels
class_names = [line.strip() for line in open("labels.txt").readlines()]

# IP Camera URL with authentication
ip_camera_url = "http://192.168.137.111:5050/shot.jpg"
username = "qwerty"
password = "123456789"

# Initialize list to store 16 frames
frames = []

while True:
    try:
        # Fetch the image from the IP camera
        response = requests.get(ip_camera_url, auth=(username, password), stream=True)
        img_array = np.array(Image.open(BytesIO(response.content)))

        # Resize the raw image into (224-height, 224-width) pixels
        image = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)

        # Convert image to float32 and normalize
        image = image.astype(np.float32) / 255.0

        # Add frame to list
        frames.append(image)

        # Keep only the last 16 frames
        if len(frames) > 16:
            frames.pop(0)

        # When we have 16 frames, predict using the model
        if len(frames) == 16:
            # Stack frames along the depth dimension to create a single input tensor
            input_data = np.stack(frames, axis=0)  # Shape: (16, 224, 224, 3)

            # Add batch dimension
            input_data = np.expand_dims(input_data, axis=0)  # Shape: (1, 16, 224, 224, 3)

            # Predict using the model
            prediction = i3d_model.predict(input_data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Print prediction and confidence score
            print("Class:", class_name, "Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

            # Clear the frames list for the next batch
            frames = []

    except Exception as e:
        print("Error fetching frame:", e)
        break
