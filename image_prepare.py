import os
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Configure logging for better debugging and tracking !
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Path to access images
image_directory = "/Users/alexandrapastouchova/Desktop/UAL/Year 3/Approaches to Machine Learning/Practical test assignment/myenv/assignment_task/Images"


# Preparing the dataset:

def data_preparation():

    # Gonna remove corrupted images before begin training
    logging.info("Checking for corrupted images...")

    for folder in os.listdir(image_directory):
        folder_path = os.path.join(image_directory, folder)
        if not os.path.isdir(folder_path):
            continue  # Skip non-directory files

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                img = image.load_img(file_path)
                img.verify()  # image is valid or not
            except (IOError, SyntaxError):
                logging.warning(f"Removing corrupted image: {file_path}")
                os.remove(file_path)

    # Augmentation and working with datasets here 
    logging.info("Loading dataset...")

    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        image_directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_data = datagen.flow_from_directory(
        image_directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    logging.info(f"Dataset loaded: {train_data.samples} training, {validation_data.samples} validation")
    return train_data, validation_data


# Building a MobileNetV2-based model

def building_model(num_classes, learning_rate=0.0005):

    logging.info("Building MobileNetV2 model...")

    # Loading MobileNetV2 
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # base layers are frozen 

    # Gonna add classification layers here 
    x = GlobalAveragePooling2D()(base_model.output)  # Making dimensionality lower
    x = Dense(128, activation='relu')(x)  # hidden layer
    x = Dense(num_classes, activation='softmax')(x)  # Output layer

    model = Model(inputs=base_model.input, outputs=x)

    # Compiling the model 
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Training the image classification model

def model_training():

    try:
        # Data
        train_data, validation_data = data_preparation()

        # Building the model
        model = building_model(num_classes=len(train_data.class_indices))

        logging.info("Starting training...")

        # training callbacks
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
        ]

        # Training the model
        model.fit(
            train_data,
            validation_data=validation_data,
            epochs=20,
            callbacks=callbacks
        )

        # Saving the model
        model.save("trained_model.keras")
        logging.info("Training complete. Model saved as trained_model.keras")

    except Exception as e:
        logging.error(f"Training failed: {e}")


# Classifing an input image using the trained model

def image_recognising(image_path):

    try:
        # Checking if the trained model exists
        if not os.path.exists("trained_model.keras"):
            logging.error("Trained model not found. Please train the model first.")
            return

        logging.info("Loading model from trained_model.keras...")
        model = tf.keras.models.load_model("trained_model.keras")

        # class labels from dataset
        class_names = sorted(os.listdir(image_directory))

        # Loading and preprocess the image
        logging.info(f"Processing image: {image_path}")
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

        # Predicting the image class
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]  
        confidence = np.max(predictions) * 100  

        print(f"Predicted class: {predicted_class_name} ({confidence:.2f}% confidence)")

    except Exception as e:
        logging.error(f"Classification failed: {e}")


# Creating interface for terminal 

if __name__ == "__main__":
    while True:  
        choice = input("Choose an action: 'train' to build the model, 'recognise' to recognise an image, or 'exit' to quit: ").strip().lower()
        
        if choice == "train":
            model_training()
        elif choice == "recognise":
            img_path = input("Enter image path: ").strip()
            if os.path.exists(img_path):
                image_recognising(img_path)
            else:
                print("Invalid file. Try again.")
        elif choice == "exit":
            print("Exiting program.")
            break
        else:
            print("Unknown command. Please enter 'train', 'recognise', or 'exit'.")