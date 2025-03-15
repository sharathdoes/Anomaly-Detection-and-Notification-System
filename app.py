import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer, Dense, Dropout, Input, GlobalAveragePooling3D, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow_hub as hub


VIDEO_LENGTH = 16  
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 8 
BATCH_SIZE = 8
EPOCHS = 20
CLASS_LABELS = ['Abuse', 'Assault', 'Burglary', 'NormalVideos','Arrest','Arson','Fighting','Shooting']


def load_videos_from_directory(directory, class_labels, video_length, img_height, img_width):
    data = []
    labels = []
    for class_label in class_labels:
        class_dir = os.path.join(directory, class_label)    
        for video_file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while len(frames) < video_length and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (img_width, img_height))
                frames.append(frame)
            cap.release()
            if len(frames) == video_length:
                data.append(np.array(frames))
                labels.append(class_labels.index(class_label))
    data = np.array(data).astype(np.float32) / 255.0  # Normalize the data
    labels = to_categorical(labels, num_classes=len(class_labels))
    return data, labels

train_dir = "Train"
test_dir = "Test"


X_train, y_train = load_videos_from_directory(train_dir, CLASS_LABELS, VIDEO_LENGTH, IMG_HEIGHT, IMG_WIDTH)
X_test, y_test = load_videos_from_directory(test_dir, CLASS_LABELS, VIDEO_LENGTH, IMG_HEIGHT, IMG_WIDTH)


class I3DWrapperLayer(Layer):
    def _init_(self, **kwargs):
        super(I3DWrapperLayer, self)._init_(**kwargs)
        self.i3d_base_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
        self.i3d_base = hub.KerasLayer(self.i3d_base_url, trainable=False)
        
    def call(self, inputs):
        return self.i3d_base(inputs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 400)


def build_i3d_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
  
    i3d_out = I3DWrapperLayer()(inputs)
    
  
    x = Dense(1024, activation='relu')(i3d_out)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

input_shape = (VIDEO_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3)
model = build_i3d_model(input_shape, NUM_CLASSES)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')


model.save("video_classification_model_i3d.h5")