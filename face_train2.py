import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
import argparse



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, user_name, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []
   
    # Training directory for the specific user
    user_dir = os.path.join(train_dir, user_name)


    if not os.path.exists(user_dir):
        raise ValueError(f"User directory '{user_name}' not found in '{train_dir}'")

    # Loop through each training image for the current user
    for person_img in os.listdir(user_dir):
        # Get the face encodings for the face in each image file
        image_path = os.path.join(user_dir, person_img)
        face = face_recognition.load_image_file(image_path)
        height, width, _ = face.shape
        face_location = (0, width, height, 0)
        face_enc = face_recognition.face_encodings(face, known_face_locations=[face_location])
        face_enc = np.array(face_enc)
        face_enc = face_enc.flatten()

        # Add face encoding for current image with corresponding label (name) to the training data
        encodings.append(face_enc)
        names.append(user_name)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(encodings, names)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train face recognition model on individual user image data.')

    parser.add_argument('user_name', type=str, help='The name of the user whose images you want to use for training')
    args = parser.parse_args()

    # STEP 1: Train the KNN classifier and save it to disk
    print(f"Training KNN classifier for user '{args.user_name}'...")
    classifier = train("TrainigData/", args.user_name,
                       model_save_path=f"TrainedModel/trained_knn_model_{args.user_name}.clf", n_neighbors=2)
    print("Training complete!")
