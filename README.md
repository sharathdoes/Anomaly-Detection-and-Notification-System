
# Anomaly Detection and Notification System

This project is designed to detect anomalies from an IP camera feed, save the anomaly frames, and send notifications via SMS using Twilio. Additionally, it serves the saved anomaly frames through a local HTTP server.

## Description

We trained a model using the I3D (Inflated 3D) architecture on the UCF Crime Dataset, which is available [here](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset). The resulting trained model (`win.h5`) is used in this project to detect anomalies in real-time from an IP camera feed.

## Features

- Capture images from an IP camera feed.
- Detect anomalies using a pre-trained deep learning model.
- Save frames with detected anomalies.
- Send SMS notifications with Twilio.
- Serve the saved anomaly frames through a local HTTP server.

## Requirements

- Python 3.10.0
- TensorFlow
- Keras
- OpenCV
- NumPy
- PIL (Pillow)
- Requests
- Twilio
- HTTP Server (http.server)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/sharathdoes/Anomaly-Detection-and-Notification-System.git
    cd anomaly-detection
    ```

2. Install the required packages:
    ```sh
    pip install tensorflow keras opencv-python numpy pillow requests twilio
    ```

3. Place your trained model (`win.h5`) and labels file (`labels.txt`) in the project directory.

## Configuration

1. **IP Camera Configuration:**
    - Update the `username`, `password`, and `ip_camera_url` variables in `class.py` with your IP camera details.

2. **Twilio Configuration:**
    - Update the `account_sid`, `auth_token`, `twilio_phone_number`, and `recipient_phone_number` variables in `class.py` with your Twilio account details and recipient's phone number.

## Usage

### Anomaly Detection Script

1. Run the anomaly detection script:
    ```sh
    python class.py
    ```

    This script will continuously fetch images from the IP camera, detect anomalies, save the anomaly frames, and send SMS notifications if an anomaly is detected.

### HTTP Server

1. Run the HTTP server script to serve the saved anomaly frames:
    ```sh
    python server.py
    ```

    This will start an HTTP server on `localhost:8000` that serves the saved anomaly frames in a user-friendly HTML table format.

## Project Structure

```
anomaly-detection/
│
├── class.py                 # Main script for detecting anomalies and sending SMS notifications
├── server.py                # Script to serve saved anomaly frames over HTTP
├── win.h5                   # Pre-trained Keras model (not included in the repo)
├── labels.txt               # Labels for the model (not included in the repo)
├── anomaly_frames/          # Directory to save anomaly frames
│
└── README.md                # Project README file
```

## Acknowledgements

- [Twilio](https://www.twilio.com/) for their API to send SMS notifications.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the machine learning framework.
- [OpenCV](https://opencv.org/) for computer vision tasks.

---

Feel free to customize this README file with additional details or sections as needed.
