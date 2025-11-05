"""
run_full_workflow.py

Combine preprocessing (upload image) and multimodal vision processing into one workflow.

Usage examples:
  # Upload a single image and process it
  python run_full_workflow.py --image "path/to/photo.jpg"

  # Process all unclassified records in Firestore
  python run_full_workflow.py --batch
"""

import os
import sys
import time
import uuid
import argparse
import tempfile
import json

# Import the multimodal processing module (it initializes Firebase at import time)
import multimodal_vision_processing as mmp

# After importing mmp, Firestore and Storage should be available via firebase_admin imports
from firebase_admin import firestore, storage


def upload_image_and_create_doc(image_path, collection_name='image_data'):
    """Upload local image to Firebase Storage and create a Firestore document.

    Returns: (doc_id, public_url, metadata)
    """
    # Normalize and attempt to locate the image in several common locations
    def _locate_image(path):
        # Expand user and normalize
        p = os.path.expanduser(path)
        p = os.path.normpath(p)
        if os.path.exists(p):
            return p

        # If a Unix-like absolute path was provivvvvded on Windows (e.g. /media/img.jpg),
        # try a repo-relative lookup by stripping leading slash
        if os.path.isabs(path) and path.startswith('/'):
            alt = os.path.join(os.getcwd(), path.lstrip('/\\'))
            alt = os.path.normpath(alt)
            if os.path.exists(alt):
                return alt

        # Look for the basename in the current working directory and ./media
        basename = os.path.basename(path)
        cand = os.path.join(os.getcwd(), basename)
        if os.path.exists(cand):
            return cand
        cand2 = os.path.join(os.getcwd(), 'media', basename)
        if os.path.exists(cand2):
            return cand2

        # As a last resort, attempt a shallow recursive search (may be a bit slow on large trees)
        for root, dirs, files in os.walk(os.getcwd()):
            if basename in files:
                return os.path.join(root, basename)

        return None

    located = _locate_image(image_path)
    if not located:
        tried = [os.path.expanduser(image_path),
                 os.path.join(os.getcwd(), image_path.lstrip('/\\')),
                 os.path.join(os.getcwd(), os.path.basename(image_path)),
                 os.path.join(os.getcwd(), 'media', os.path.basename(image_path))]
        tried = [os.path.normpath(t) for t in tried]
        raise FileNotFoundError(
            f"Image not found: {image_path}\nTried paths:\n  " + "\n  ".join(tried) +
            "\nYou can pass a correct Windows path (e.g. C:\\path\\to\\image.jpg) or place the file in the project 'media' folder."
        )
    image_path = located

    bucket = storage.bucket()
    filename = os.path.basename(image_path)
    dest_path = f"images/{uuid.uuid4().hex}_{filename}"

    # Upload file
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(image_path)
    try:
        blob.make_public()
    except Exception:
        # Not all projects allow public URLs; ignore if it fails
        pass

    public_url = getattr(blob, 'public_url', None) or blob.self_link

    # Prepare document
    doc_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    metadata = {
        "caption": "",
        "source": filename,
        "timestamp": timestamp,
        "xy_coordinate": [0, 0]
    }

    document_data = {
        "image_classification": {
            "hazards": "",
            "health_risk": "",
            "logistics": "",
            "people": ""
        },
        "image_path": public_url,
        "metadata": metadata
    }

    db = firestore.client()
    db.collection(collection_name).document(doc_id).set(document_data)

    print(f"Uploaded and created doc: {doc_id} -> {public_url}")
    return doc_id, public_url, metadata


def process_single_doc(doc_id, collection_name='image_data'):
    """Fetch a document by id, run the vision LLM, and update the classification field."""
    db = firestore.client()
    doc_ref = db.collection(collection_name).document(doc_id)
    doc = doc_ref.get()
    if not doc.exists:
        print(f"Document {doc_id} not found.")
        return False

    rec = doc.to_dict()
    image_path = rec.get('image_path')
    metadata = rec.get('metadata', {})

    if not image_path:
        print(f"Document {doc_id} has no image_path. Skipping.")
        return False

    print(f"Processing document {doc_id} with image {image_path}")

    try:
        signed_url = mmp.get_signed_url_from_storage_path(image_path)
        lat = metadata.get('latitude') or metadata.get('xy_coordinate', [0, 0])[0]
        lon = metadata.get('longitude') or metadata.get('xy_coordinate', [0, 0])[1]
        ts = metadata.get('timestamp', "")

        result = mmp.process_image_with_llm(signed_url, metadata, lat, lon, ts)

        # Update Firestore
        mmp.update_firestore_classification(doc_id, result, collection_name)
        print(f"Updated classification for {doc_id}")
        return True

    except Exception as e:
        print(f"Error processing {doc_id}: {e}")
        return False


def process_unclassified_batch(collection_name='image_data', limit=None):
    """Fetch unclassified records and process them sequentially."""
    records = mmp.fetch_firestore_image_records(collection_name)
    to_process = []
    for doc_id, rec in records.items():
        existing = rec.get('image_classification', {})
        if not existing or not any(v for v in existing.values() if v):
            to_process.append(doc_id)

    if limit:
        to_process = to_process[:limit]

    print(f"Found {len(to_process)} unclassified documents to process.")
    processed = 0
    errors = 0
    for doc_id in to_process:
        ok = process_single_doc(doc_id, collection_name)
        if ok:
            processed += 1
        else:
            errors += 1

    print(f"Batch summary: processed={processed}, errors={errors}, total={len(to_process)}")


def _parse_args():
    p = argparse.ArgumentParser(description='Combined upload + vision processing workflow')
    p.add_argument('--image', help='Path to a local image to upload and process')
    p.add_argument('--doc-id', help='Process a specific Firestore document by id')
    p.add_argument('--batch', action='store_true', help='Process unclassified documents in Firestore')
    p.add_argument('--collection', default='image_data', help='Firestore collection name')
    p.add_argument('--limit', type=int, help='Limit number of documents to process in batch')
    return p.parse_args()


def main():
    args = _parse_args()

    if args.image:
        doc_id, url, meta = upload_image_and_create_doc(args.image, args.collection)
        # Process the newly created doc
        process_single_doc(doc_id, args.collection)
        return

    if args.doc_id:
        process_single_doc(args.doc_id, args.collection)
        return

    if args.batch:
        process_unclassified_batch(args.collection, limit=args.limit)
        return

    print("No action specified. Use --image, --doc-id, or --batch. Use -h for help.")


if __name__ == '__main__':
    main()
