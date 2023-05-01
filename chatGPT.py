import tensorflow as tf
import numpy as np
import cv2
import keras.backend

# Load the pre-trained FaceNet model
model = tf.keras.models.load_model("facenet_keras.h5")

# Define a function to preprocess the input image
def preprocess_image(image):
    # Resize the image to (160, 160) and normalize the pixel values
    image = cv2.resize(image, (160, 160))
    image = image / 255.0
    # Convert the image to a 4D tensor
    image = np.expand_dims(image, axis=0)
    return image

# Define a function to extract the face embeddings from an image
def get_face_embeddings(image):
    # Preprocess the image
    image = preprocess_image(image)
    # Use the FaceNet model to extract the face embeddings
    embeddings = model.predict(image)
    return embeddings

# Define a function to calculate the Euclidean distance between two embeddings
def distance(embedding1, embedding2):
    diff = np.subtract(embedding1, embedding2)
    dist = np.sum(np.square(diff))
    return dist

# Define a function to recognize a face in an image
def recognize_face(image, known_embeddings, threshold=0.5):
    # Extract the face embeddings from the image
    query_embedding = get_face_embeddings(image)
    # Loop through the known embeddings and calculate the distance
    min_distance = 999
    identity = None
    for name, embedding in known_embeddings.items():
        dist = distance(query_embedding, embedding)
        if dist < min_distance:
            min_distance = dist
            identity = name
    # Check if the distance is below the threshold
    if min_distance < threshold:
        return identity
    else:
        return "Unknown"
