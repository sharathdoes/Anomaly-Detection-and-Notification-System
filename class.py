import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from io import BytesIO
from PIL import Image
import time
from twilio.rest import Client
import os

np.set_printoptions(suppress=True)

custom_objects = {'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}

# Load the model
try:
    model = load_model("win.h5", custom_objects=custom_objects, compile=False)
except ValueError as e:
    print(f"Error loading model: {e}")
    exit()

# Load labels
try:
    with open("labels.txt", "r") as file:
        class_names = [line.strip() for line in file.readlines()]
except FileNotFoundError:
    print("Labels file not found.")
    exit()

username = 'qwerty'
password = '123456789'
ip_camera_url = "http://192.168.137.160:5050/shot.jpg"

# Twilio credentials
account_sid = 'ACcb6c4a758ccb55ba50d310c4c52ace1c'
auth_token = '36d0108d7c307af8b0ed5d24a02893d7'
twilio_phone_number = '+15752317745'
recipient_phone_number = '+919391631714'

# Initialize the Twilio Client
client = Client(account_sid, auth_token)

def get_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        response.raise_for_status()
        data = response.json()
        location = data['loc']  # loc format is "latitude,longitude"
        return location
    except requests.exceptions.RequestException as e:
        print(f"Failed to get location: {e}")
        return "unknown location"

# Create the directory for anomaly frames if it doesn't exist
anomaly_frames_dir = "anomaly_frames"
if not os.path.exists(anomaly_frames_dir):
    os.makedirs(anomaly_frames_dir)

while True:
    try:
        response = requests.get(ip_camera_url, auth=HTTPBasicAuth(username, password), timeout=10)
        response.raise_for_status()
        img_array = np.array(Image.open(BytesIO(response.content)))
    except requests.exceptions.RequestException as e:
        print(f"Failed to grab frame: {e}")
        time.sleep(3)  # Retry after a short delay
        continue

    image = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("IP Camera Image", image)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    image = (image / 127.5) - 1

    # Predict
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    if class_name.strip().lower() != "normal":
        location = get_location()
        message_body = f'Detected anomaly is {class_name.strip()}. Location: {location} http://10.45.23.47:8000/'

        # Sending the SMS
        message = client.messages.create(
            body=message_body,
            from_=twilio_phone_number,
            to=recipient_phone_number
        )

        # Print the message SID (unique identifier for the message)
        print(f"Message sent successfully! Message SID: {message.sid}")

        # Save the frame to the anomaly_frames directory
        frame_filename = os.path.join(anomaly_frames_dir, f"anomaly_{int(time.time())}.jpg")
        cv2.imwrite(frame_filename, img_array)
        print(f"Anomaly frame saved as: {frame_filename}")

    print("Class:", class_name.strip(), "Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()