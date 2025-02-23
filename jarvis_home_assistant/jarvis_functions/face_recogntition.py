import face_recognition
import os
import cv2
import numpy as np
import math


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value)) + '%'


def load_known_faces(face_dir='faces'):
    """Loads and encodes faces from the specified directory."""
    known_face_encodings = []
    known_face_names = []

    for image_name in os.listdir(face_dir):
        image_path = os.path.join(face_dir, image_name)
        face_image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(face_image)

        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.splitext(image_name)[0])
        else:
            print(f"Warning: No face found in {image_name}")

    return known_face_encodings, known_face_names


def recognize_face():
    """Captures video from webcam and recognizes known faces. Returns the first recognized face's name and stops."""
    known_face_encodings, known_face_names = load_known_faces()
    video_capture = cv2.VideoCapture(1)

    if not video_capture.isOpened():
        print("Error: Video source not found...")
        return None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])
                video_capture.release()
                cv2.destroyAllWindows()
                return f'{name} ({confidence})'

        if cv2.waitKey(1) == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return None
