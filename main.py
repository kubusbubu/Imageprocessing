import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from PIL import ImageFile
import numpy as np
import os
import random

# This line helps to avoid image truncation errors
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Define paths to the train and test directories
train_dir = 'TRAIN'
test_dir = 'TEST'

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

# Adjust steps per epoch and validation steps
steps_per_epoch = train_generator.n // train_generator.batch_size // 2
validation_steps = test_generator.n // test_generator.batch_size // 2

# Building the model, maybe add
model = Sequential([
    Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  # 7 classes for 7 yoga poses
])

# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training the model
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(
    train_generator,
    callbacks=[early_stop],
    steps_per_epoch=steps_per_epoch,  # adjusted based on your dataset size
    epochs=15,
    validation_data=test_generator,
    validation_steps=validation_steps  # adjusted based on your dataset size
)

# Evaluating the model on test data
test_loss, test_acc = model.evaluate(test_generator, steps=validation_steps)
print(f'Test accuracy: {test_acc}')

def predict_pose(img_path):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    preprocessed_img = img_array_expanded_dims / 255.  # Rescale the image

    # Make prediction
    predictions = model.predict(preprocessed_img)
    predicted_class = np.argmax(predictions, axis=-1)[0]

    # Get the class label
    predicted_label = class_labels[predicted_class]
    print(f'The predicted pose for image {os.path.basename(img_path)} in {os.path.dirname(img_path)} is: {predicted_label}')

# Get the class labels from the training data
class_labels = train_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}

# Path to the test folder
test_folder_path = 'TEST'

# Get list of image paths from the test folder classes
image_paths = []
for class_folder in os.listdir(test_folder_path):
    class_folder_path = os.path.join(test_folder_path, class_folder)

    images = os.listdir(class_folder_path)
    three_random = [random.choice(images) for i in range(3)]
    for img_file in three_random:
        image_paths.append(os.path.join(class_folder_path, img_file))


# Test a few images from the test folder classes
for img_path in image_paths:  # Adjust this slice to test more or fewer images
    predict_pose(img_path)

