"""
dump_firestore_image_data.py

Script to dump all documents from Firestore 'image_data' collection to a local JSON file.
"""

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from config import FIREBASE_CONFIG

# --- Firebase Initialization ---
cred = credentials.Certificate("jamaica-realtime-crisis-map-firebase-adminsdk-fbsvc-2b8d724ab7.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': FIREBASE_CONFIG['storageBucket'],
    'projectId': FIREBASE_CONFIG['projectId']
})

def dump_firestore_collection(collection_name='image_data', output_file='firestore_image_data_dump.json'):
    db = firestore.client()
    docs = db.collection(collection_name).stream()
    data = {doc.id: doc.to_dict() for doc in docs}

    # Custom serializer for Firestore types
    def firestore_serializer(obj):
        # Firestore Timestamp
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        # Firestore DocumentReference
        if hasattr(obj, 'path'):
            return obj.path
        # Fallback: string representation
        return str(obj)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=firestore_serializer)
    print(f"Dumped {len(data)} documents to {output_file}")

if __name__ == "__main__":
    dump_firestore_collection()
