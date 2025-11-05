import cv2
import os
import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime
import json
import uuid
import tempfile
import sys
from config import FIREBASE_CONFIG

# --- CONFIGURATION ---
def get_video_path():
    # Allow a path to be passed as the first CLI argument for non-interactive use
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        candidate = os.path.expanduser(candidate)
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            return candidate
        print(f"Error: CLI path '{candidate}' not found.")

    # Otherwise prompt the user (works on Windows and Unix). Normalizes the path to avoid \n unicode-escape issues.
    while True:
        #video_path = input("Please enter the path to your video file (or 'q' to quit): ")
        video_path = video_path
        if video_path.lower() == 'q':
            exit()
        video_path = os.path.expanduser(video_path)
        video_path = os.path.normpath(video_path)
        if os.path.exists(video_path):
            return video_path
        print(f"Error: File '{video_path}' not found. Please try again.")

# Video processing settings
interval = 5  # seconds between frames

# Get video path from user
video_path = get_video_path()
#video_path = 'input_video_3.mp3'

# Firebase config
cred = credentials.Certificate("jamaica-realtime-crisis-map-firebase-adminsdk-fbsvc-2b8d724ab7.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': FIREBASE_CONFIG['storageBucket'],
    'projectId': FIREBASE_CONFIG['projectId']
})

# Initialize Firebase services
db = firestore.client()
bucket = storage.bucket()

# --- PROCESS VIDEO ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * interval)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps:.2f}, Total Frames: {frame_count}")

frame_number = 0
saved_count = 0

# JSON metadata list
frames_metadata = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number % frame_interval == 0:
        timestamp = datetime.utcnow().isoformat() + "Z"
        filename = f"frame_{saved_count:05d}.jpg"

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            temp_path = tmp_file.name
            
        # Save frame to temporary file
        cv2.imwrite(temp_path, frame)

        try:
            # Upload to Firebase Storage in the images folder
            blob = bucket.blob(f"images/{filename}")
            blob.upload_from_filename(temp_path)
            blob.make_public()  # Make public URL for access
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Generate a unique document ID
        doc_id = str(uuid.uuid4())

        # Prepare document data according to the schema
        document_data = {
            "image_classification": {
                "hazards": "",
                "health_risk": "",
                "logistics": "",
                "people": ""
            },
            "image_path": blob.public_url,
            "metadata": {
                "caption": "",
                "source": os.path.basename(video_path),
                "timestamp": timestamp,
                "xy_coordinate": [0, 0]  # Default coordinates, update as needed
            }
        }

        # Save to Firestore
        db.collection('image_data').document(doc_id).set(document_data)

        print(f"Uploaded frame {saved_count} -> {blob.public_url}")
        print(f"Created document with ID: {doc_id}")
        saved_count += 1

    frame_number += 1

cap.release()
print(f"✅ Done! Uploaded {saved_count} frames to Firebase Storage and database.")
