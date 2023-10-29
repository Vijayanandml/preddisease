import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

    # Define the paths to the image folders
non_disease_folder = r"C:\Users\Lenovo\Documents\ICH11\Dataset\nondisease"  # Replace with the path to your non-disease image folder
disease_folder = r"C:\Users\Lenovo\Documents\ICH11\Dataset\disease"   # Replace with the path to your disease image folder

# Define the output labels
labels = {'non_disease': 0, 'disease': 1}

# Define the input shape and preprocessing function
input_shape = (224, 224, 3)

# Define the path to the ResNet50 weights file
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the images and labels
X = []
y = []

# Load disease images
for filename in os.listdir(disease_folder)[:10000]:
    img_path = os.path.join(disease_folder, filename)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)

    X.append(img_array)
    y.append(labels['disease'])

# Load non-disease images
for filename in os.listdir(non_disease_folder)[:10000]:
    img_path = os.path.join(non_disease_folder, filename)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)

    X.append(img_array)
    y.append(labels['non_disease'])

# Convert lists to arrays
X = np.array(X)
y = np.array(y)

# Convert labels to one-hot encoded vectors
y = tf.keras.utils.to_categorical(y, num_classes=2)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
num_epochs = 3
batch_size = 8
epoch=34
num_array=y_train =train

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Shuffle the training data
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Training
    train_loss = 0.0
    train_accuracy = 0.0
    num_batches = 0

    for i in range(0, len(X_train), batch_size):
        batch_images = X_train[i:i+batch_size]
        batch_labels = y_train[i:i+batch_size]

        with tf.GradientTape() as tape:
            logits = model(batch_images, training=True)
            loss_value = tf.keras.losses.categorical_crossentropy(batch_labels, logits)
            
        # Backward pass and update weights
        gradients = tape.gradient(loss_value, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss += loss_value.numpy().mean()

        train_accuracy += np.mean(np.argmax(logits, axis=1) == np.argmax(batch_labels, axis=1))
        num_batches += 1

    train_loss /= num_batches
    train_accuracy /= num_batches

    # Validation
    val_loss = 0.0
    val_accuracy = 0.0
    num_batches = 0

    for i in range(0, len(X_test), batch_size):
        batch_images = X_test[i:i+batch_size]
        batch_labels = y_test[i:i+batch_size]

        logits = model(batch_images, training=False)
        loss_value = tf.keras.losses.categorical_crossentropy(batch_labels, logits)

        val_loss += loss_value.numpy().mean()
        val_accuracy += np.mean(np.argmax(logits, axis=1) == np.argmax(batch_labels, axis=1))
        num_batches += 1

    val_loss /= num_batches
    val_accuracy /= num_batches

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save the trained model
model.save('trained_model.h5')

# Load the trained model
model = load_model('trained_model.h5')

# Define the labels
labels = {0: 'nondisease', 1: 'disease'}

# Function to classify an input image
def classify_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (96, 96))  # Resize to match the model's input size
    img = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img, axis=0))

    # Make predictions
    predictions = model.predict(img)

    # Get the predicted class label
    predicted_class = np.argmax(predictions)

    # Get the class name
    class_name = labels[predicted_class]

    return class_name

# Input image path
input_image_path = "C:/Users/Lenovo/Documents/ICH11/tld.jpg"  # Replace with the path to your input image

# Classify the input image
result = classify_image(input_image_path)
print(f'The image is classified as: {result}')