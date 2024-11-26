import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize the video capture object for the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load known faces
known_face_encodings = []
known_face_names = []
for person in ["images", "Souvik"]:
    image_path = f"./faces/{person}.jpg"
    image = face_recognition.load_image_file(image_path)
    
 # Ensure that the encoding step doesn't fail
    try:
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(person)
    except IndexError:
        print(f"Could not encode the face in {image_path}. Check the image file.")
    
# Create a copy of the known face names for expected students
expected_students = known_face_names.copy()

# Initialize variables for face locations and encodings
face_locations = []
face_encodings = []

# Get the current time
now = datetime.now()
current_date = now.strftime("%d-%m-%y")

# Open a CSV file for writing attendance
with open(f"{current_date}.csv", "w+", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Time"])

    # Start the main video capture loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break
        
        # Resize and convert frame for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Recognize faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Display presence on the video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner_of_text = (10, 100)
                font_scale = 1.5
                font_color = (255, 0, 0)
                thickness = 3
                line_type = 2
                cv2.putText(frame, f"{name} PRESENT", bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)

                if name in expected_students:
                    expected_students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    writer.writerow([name, current_time])

        # Display the frame with attendance information
        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
