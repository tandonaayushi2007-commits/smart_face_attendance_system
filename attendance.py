"""
Attendance System - Face Recognition Module
Fixed triple-quoted string issue.
"""

import cv2
import face_recognition
import os
import pickle
import numpy as np
from datetime import datetime

DATASET = "dataset/"
MODEL = "encodings.pkl"
LOG_FILE = "update_log.txt"


def load_encodings():
    """Load stored encodings if file exists."""
    if os.path.exists(MODEL):
        return pickle.load(open(MODEL, "rb"))
    return {"encodings": [], "names": []}


def encode_faces():
    """Encode all faces from the dataset and save them."""
    encodings = []
    names = []

    for student in os.listdir(DATASET):
        folder = os.path.join(DATASET, student)

        if not os.path.isdir(folder):
            continue

        for img in os.listdir(folder):
            image_path = os.path.join(folder, img)
            image = face_recognition.load_image_file(image_path)
            boxes = face_recognition.face_locations(image)

            if len(boxes) == 0:
                print(f"❌ No face found in {image_path}")
                continue

            encoding = face_recognition.face_encodings(image)[0]

            encodings.append(encoding)
            names.append(student)

    data = {"encodings": encodings, "names": names}
    pickle.dump(data, open(MODEL, "wb"))
    print("✅ Encodings updated successfully!")

    return data


def check_duplicate_face(new_encoding, data, threshold=0.45):
    """
    Checks whether a new captured face is already in the dataset.
    threshold: lower = more strict
    """
    if len(data["encodings"]) == 0:
        return False

    matches = face_recognition.compare_faces(data["encodings"], new_encoding, tolerance=threshold)
    return True if True in matches else False


def update_faces(student_name, new_image):
    """Add new image + prevent duplicates + retrain automatically."""

    folder = os.path.join(DATASET, student_name)

    # Create folder if not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Compute encoding of new image
    rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)

    if len(boxes) == 0:
        print("❌ No face detected. Update cancelled.")
        return

    new_encoding = face_recognition.face_encodings(rgb)[0]

    # Load existing encodings
    data = load_encodings()

    # Check if duplicate
    if check_duplicate_face(new_encoding, data):
        print("⚠️ Duplicate face detected! Not adding again.")
        return

    # Save new image
    count = len(os.listdir(folder))
    img_path = os.path.join(folder, f"{count}.jpg")
    cv2.imwrite(img_path, new_image)

    # Log update
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} - Added new image for {student_name}\n")

    print("🟢 New face added. Updating model...")

    # Retrain encodings
    encode_faces()
