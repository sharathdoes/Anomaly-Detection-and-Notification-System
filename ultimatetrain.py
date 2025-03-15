import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
from tensorflow.keras.layers import Layer, Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, RandomGamma, HueSaturationValue
from albumentations import ImageOnlyTransform

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 8
CLASS_LABELS = ['Abuse', 'Assault', 'Burglary', 'NormalVideos', 'Arrest', 'Arson', 'Fighting', 'Shooting']

# Data Augmentation class using albumentations
class CustomAlbumentationTransform(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(CustomAlbumentationTransform, self).__init__(always_apply, p)
        self.aug = Compose([
            HorizontalFlip(),  # Randomly flip the image horizontally
            VerticalFlip(),    # Randomly flip the image vertically
            RandomBrightnessContrast(),  # Randomly change brightness and contrast
            RandomGamma(),  # Randomly change gamma
            HueSaturationValue()  # Randomly change hue, saturation, and value
        ])

    def apply(self, img, **params):
        return self.aug(image=img)["image"]

# Apply augmentation to a video (list of frames)
def augment_video(frames):
    transform = CustomAlbumentationTransform(p=1.0)
    augmented_frames = [transform.apply(frame) for frame in frames]
    return np.array(augmented_frames)

# Load videos from a directory, augment them if specified
def load_videos_from_directory(directory, class_labels, img_height, img_width, augment=False):
    data = []
    labels = []
    for class_label in class_labels:
        class_dir = os.path.join(directory, class_label)
        for video_file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (img_width, img_height))  # Resize frame
                frames.append(frame)
            cap.release()
            if len(frames) > 0:
                if augment:
                    frames = augment_video(frames)  # Apply augmentation
                data.append(np.array(frames))
                labels.append(class_labels.index(class_label))
    
    labels = to_categorical(labels, num_classes=len(class_labels))  # Convert labels to one-hot encoding
    return data, labels

# Load train and test data
train_dir = "Train"
test_dir = "Test"
X_train, y_train = load_videos_from_directory(train_dir, CLASS_LABELS, IMG_HEIGHT, IMG_WIDTH, augment=True)
X_test, y_test = load_videos_from_directory(test_dir, CLASS_LABELS, IMG_HEIGHT, IMG_WIDTH, augment=False)

# Custom layer for I3D feature extraction
class I3DWrapperLayer(Layer):
    def __init__(self, trainable=True, **kwargs):
        super(I3DWrapperLayer, self).__init__(**kwargs)
        self.i3d_base_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
        self.i3d_base = hub.KerasLayer(self.i3d_base_url, trainable=trainable)
        
    def call(self, inputs):
        return self.i3d_base(inputs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 400)

# Extract features from videos using I3D model
def extract_features(data, img_height, img_width):
    i3d_layer = I3DWrapperLayer()
    features = []
    for frames in data:
        frames = np.array(frames) / 255.0  # Normalize frames
        frames = np.expand_dims(frames, axis=0)  # Add batch dimension
        features.append(i3d_layer(frames).numpy())
    return features

# Extract features from train and test data
X_train_features = extract_features(X_train, IMG_HEIGHT, IMG_WIDTH)
X_test_features = extract_features(X_test, IMG_HEIGHT, IMG_WIDTH)

# Find the maximum sequence length
max_len = max(max(len(seq) for seq in X_train_features), max(len(seq) for seq in X_test_features))

# Pad sequences to ensure uniform length
X_train_padded = pad_sequences(X_train_features, maxlen=max_len, dtype='float32', padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_features, maxlen=max_len, dtype='float32', padding='post', truncating='post')

# Build and train LSTM model with regularization and advanced scheduling
def build_lstm_model(hp):
    inputs = Input(shape=(None, 400))  # Variable-length sequences, feature dimension 400
    
    x = LSTM(hp.Int('units', min_value=256, max_value=1024, step=256), return_sequences=True)(inputs)
    x = BatchNormalization()(x)  # Add batch normalization
    x = Dropout(hp.Float('dropout', min_value=0.3, max_value=0.7, step=0.1))(x)  # Add dropout
    x = LSTM(hp.Int('units_2', min_value=128, max_value=512, step=128))(x)
    x = BatchNormalization()(x)  # Add batch normalization
    x = Dropout(hp.Float('dropout_2', min_value=0.3, max_value=0.7, step=0.1))(x)  # Add dropout
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer with softmax activation
    
    model = Model(inputs, outputs)
    
    model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter tuning using RandomSearch
tuner = RandomSearch(build_lstm_model, objective='val_accuracy', max_trials=20, executions_per_trial=3, directory='tuner', project_name='video_classification')

# Conduct a search for the best hyperparameters
tuner.search(X_train_padded, y_train, epochs=20, validation_data=(X_test_padded, y_test))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_callback = LearningRateScheduler(scheduler)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the best model with learning rate scheduling and early stopping
best_model.fit(X_train_padded, y_train, epochs=100, batch_size=8, validation_data=(X_test_padded, y_test), callbacks=[lr_callback, reduce_lr, early_stopping])

# Evaluate and save the model
loss, accuracy = best_model.evaluate(X_test_padded, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

best_model.save("video_classification_model_lstm_best.h5")
