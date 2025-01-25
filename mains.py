import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# Load the FaceNet model (pretrained on VGGFace2 dataset)
def load_facenet_model():
    model_path = tf.keras.utils.get_file(
        "facenet_keras.h5",
        "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_keras.h5",
        cache_subdir="models"
    )
    model = load_model(model_path)
    return model

# Preprocess the image for FaceNet
def preprocess_image(image_path, target_size=(160, 160)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=0)
    img = (img - 127.5) / 127.5  # Normalize pixel values
    return img

# Generate embedding for an image
def get_embedding(model, image_path):
    processed_image = preprocess_image(image_path)
    embedding = model.predict(processed_image)
    return embedding[0]

# Recognize faces
def recognize_faces(model, known_faces, known_labels, test_image_path):
    test_embedding = get_embedding(model, test_image_path)
    similarities = cosine_similarity([test_embedding], known_faces)
    max_sim_index = np.argmax(similarities)
    max_similarity = similarities[0][max_sim_index]

    if max_similarity > 0.8:  # Similarity threshold
        return known_labels[max_sim_index], max_similarity
    else:
        return "Unknown", max_similarity

# Main code
if __name__ == "__main__":
    # Load the model
    print("Loading FaceNet model...")
    facenet_model = load_facenet_model()

    # Create a dataset of known faces
    known_faces_dir = "known_faces"  # Directory containing labeled images
    known_faces = []
    known_labels = []

    for label in os.listdir(known_faces_dir):
        label_dir = os.path.join(known_faces_dir, label)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            embedding = get_embedding(facenet_model, img_path)
            known_faces.append(embedding)
            known_labels.append(label)

    known_faces = np.array(known_faces)
    print("Known faces embeddings generated.")

    # Test recognition
    test_image_path = "test_image.jpg"  # Path to the test image
    result, similarity = recognize_faces(facenet_model, known_faces, known_labels, test_image_path)
    print(f"Result: {result}, Similarity: {similarity}")
