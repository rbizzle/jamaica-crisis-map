"""
multimodal_vision_processing.py

Helper utilities to process images with a vision-enabled LLM and return structured JSON
suitable for your Firestore `image_data` schema.
"""

import os
import json
import time
import requests
import warnings
import argparse
from langchain_core.messages import HumanMessage

# Prefer the standalone langchain-openai package
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

# Suppress known deprecation warnings
warnings.filterwarnings("ignore", message=r".*was deprecated.*")

# Firebase initialization
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Optional geocoding (generate lat/lon from on-screen location description)
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Simple in-memory cache for geocoding results to avoid repeated queries
GEOCODE_CACHE = {}


# ==========================================================
# 🔹 FIREBASE CONFIG - Handles both local and Streamlit Cloud
# ==========================================================
def get_firebase_config():
    """Get Firebase config from Streamlit secrets or local config file."""
    try:
        import streamlit as st
        return {
            'storageBucket': st.secrets["firebase"]["storageBucket"],
            'projectId': st.secrets["firebase"]["projectId"]
        }
    except:
        # Fallback for local development
        try:
            from config import FIREBASE_CONFIG
            return FIREBASE_CONFIG
        except ImportError:
            # Last resort hardcoded values
            return {
                'storageBucket': 'jamaica-realtime-crisis-map.firebasestorage.app',
                'projectId': 'jamaica-realtime-crisis-map'
            }

FIREBASE_CONFIG = get_firebase_config()


def geocode_location(query, delay=1.0):
    """Geocode a free-text location string using OpenStreetMap Nominatim.

    Returns (lat, lon) tuple or None on failure. Caches results in-memory.
    """
    if not query:
        return None
    key = str(query).strip().lower()
    if key in GEOCODE_CACHE:
        return GEOCODE_CACHE[key]
    try:
        geolocator = Nominatim(user_agent="crisis_map_geocoder")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=delay, max_retries=2)
        loc = geocode(query)
        if loc:
            coords = (float(loc.latitude), float(loc.longitude))
            GEOCODE_CACHE[key] = coords
            return coords
    except Exception as e:
        print(f"  ⚠️ Geocode error for '{query}': {e}")
        return None
    return None


# ==========================================================
# 🔹 SYSTEM PROMPT
# ==========================================================
SYSTEM_PROMPT = """You are a disaster-response vision analyst.
Analyze the image and return valid JSON exactly in this format 
    - if there is no GPS latitude or longitude use onscreen location desription if present  
    - if there is no timestamp info use onscreen timestamp if present
    (no extra prose):

{
  "hazards": [],
  "logistics": [],
  "health_risks": [],
  "people": [{"count": 0, "activity": "", "distress_signs": false}],
  "location": {"lat": 0, "lon": 0, "location_description: ""}
  "timestamp": "" or timestamp info onscreen
}

Only return valid JSON with no additional text."""


# ==========================================================
# 🔹 FIREBASE INITIALIZATION
# ==========================================================
def initialize_firebase():
    """Initialize Firebase only once, with proper error handling for both local and cloud."""
    if firebase_admin._apps:
        print("Firebase already initialized, skipping...")
        return
    
    try:
        # Try Streamlit secrets first (for cloud deployment)
        try:
            import streamlit as st
            if 'firebase_credentials' in st.secrets:
                print("Using Firebase credentials from Streamlit secrets...")
                service_account_info = dict(st.secrets["firebase_credentials"])
                cred = credentials.Certificate(service_account_info)
            else:
                raise KeyError("No firebase_credentials in secrets")
        except (ImportError, KeyError):
            # Fallback to local JSON file (for local development)
            print("Using local service account JSON file...")
            key_file = "jamaica-realtime-crisis-map-firebase-adminsdk-fbsvc-2b8d724ab7.json"
            
            if not os.path.exists(key_file):
                raise FileNotFoundError(f"Service account key not found: {key_file}")
            
            # Verify JSON is valid before using it
            with open(key_file, 'r') as f:
                service_account_info = json.load(f)
            
            cred = credentials.Certificate(service_account_info)
        
        firebase_admin.initialize_app(cred, {
            'storageBucket': FIREBASE_CONFIG['storageBucket'],
            'projectId': FIREBASE_CONFIG['projectId']
        })
        print("✓ Firebase initialized successfully")
        
    except Exception as e:
        print(f"✗ Firebase initialization failed: {e}")
        raise

# Initialize Firebase once at module load
initialize_firebase()


# ==========================================================
# 🔹 FIREBASE STORAGE UTILITIES
# ==========================================================
def get_signed_url_from_storage_path(image_path, expiration_minutes=60):
    """
    Convert a Firebase Storage path or URL to a signed URL.
    Handles both gs:// URLs and public Firebase download URLs.
    """
    try:
        bucket = storage.bucket()

        # Normalize blob path
        if image_path.startswith('gs://'):
            path_parts = image_path.replace('gs://', '').split('/', 1)
            blob_path = path_parts[1] if len(path_parts) > 1 else path_parts[0]
        elif 'storage.googleapis.com' in image_path or 'firebasestorage.app' in image_path:
            import urllib.parse
            if '/o/' in image_path:
                blob_path = image_path.split('/o/')[1].split('?')[0]
                blob_path = urllib.parse.unquote(blob_path)
            else:
                parts = image_path.split('.com/')
                blob_path = parts[1].split('?')[0] if len(parts) > 1 else image_path
        else:
            blob_path = image_path

        print(f"  Extracted blob path: {blob_path}")

        blob = bucket.blob(blob_path)
        if not blob.exists():
            print(f"  ⚠️ Warning: Blob not found. Making public instead...")
            blob.make_public()
            return blob.public_url

        # Generate signed URL (valid for X minutes)
        return blob.generate_signed_url(
            expiration=time.time() + (expiration_minutes * 60),
            version='v4'
        )

    except Exception as e:
        print(f"  ⚠️ Error generating signed URL: {e}")
        return image_path


# ==========================================================
# 🔹 FIRESTORE HELPERS
# ==========================================================
def fetch_firestore_image_records(collection_name='image_data'):
    """Fetch all documents from a Firestore collection."""
    client = firestore.client()
    return {d.id: d.to_dict() for d in client.collection(collection_name).stream()}


def update_firestore_classification(doc_id, classification, collection_name='image_data'):
    """Update the image_classification field for a specific Firestore document."""
    client = firestore.client()
    client.collection(collection_name).document(doc_id).update({
        'image_classification': classification
    })


def is_classified(classification: dict) -> bool:
    """Return True if the classification dict contains meaningful (non-placeholder) data.

    Heuristics:
    - non-empty hazards/logistics/health_risks lists or non-empty strings
    - any people entry with count > 0, non-empty activity, or distress_signs True
    - location lat/lon not both zero
    - non-empty timestamp string
    """
    if not classification or not isinstance(classification, dict):
        return False

    # Check list/string fields
    for key in ('hazards', 'logistics', 'health_risks'):
        val = classification.get(key)
        if isinstance(val, list) and len(val) > 0:
            return True
        if isinstance(val, str) and val.strip():
            return True

    # Check people observations
    people = classification.get('people')
    if isinstance(people, list):
        for p in people:
            if not isinstance(p, dict):
                continue
            # count
            try:
                cnt = int(p.get('count', 0))
            except Exception:
                cnt = 0
            if cnt > 0:
                return True
            # distress signs
            if p.get('distress_signs'):
                return True
            # activity text
            act = p.get('activity')
            if isinstance(act, str) and act.strip():
                return True

    # Check location coordinates
    loc = classification.get('location')
    if isinstance(loc, dict):
        try:
            lat = float(loc.get('lat', 0) or 0)
            lon = float(loc.get('lon', 0) or 0)
            if lat != 0 or lon != 0:
                return True
        except Exception:
            pass

    # Timestamp
    ts = classification.get('timestamp')
    if isinstance(ts, str) and ts.strip():
        return True

    return False


# ==========================================================
# 🔹 LLM PROCESSING
# ==========================================================
def process_image_with_llm(image_url, metadata, latitude=None, longitude=None, timestamp=None, max_retries=3):
    """
    Sends image URL + metadata to the LLM and returns parsed JSON dict.
    Uses GPT-4o (vision-capable).
    """
    api_key = os.getenv("OPENAI_API_KEY") or input("Enter your OPENAI_API_KEY: ").strip()
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY provided")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=api_key,
        max_tokens=1000
    )

    lat = latitude or metadata.get("latitude", 0)
    lon = longitude or metadata.get("longitude", 0)
    location_info  = '' or metadata.get("location_description", "")
    ts = timestamp or metadata.get("timestamp", "")

    user_content = f"""Analyze this disaster response image.

Metadata: {json.dumps(metadata, indent=2)}
Location: lat={lat}, lon={lon}
Timestamp: {ts}

Return ONLY valid JSON in the specified format."""

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}: Calling GPT-4o vision API...")

            message = HumanMessage(content=[
                {"type": "text", "text": SYSTEM_PROMPT + "\n\n" + user_content},
                {"type": "image_url", "image_url": {"url": image_url}}
            ])

            response = llm.invoke([message])
            text = response.content if hasattr(response, 'content') else str(response)

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Extract JSON substring if model added extra text
                start, end = text.index("{"), text.rindex("}") + 1
                return json.loads(text[start:end])

        except Exception as e:
            print(f"  ✗ Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                sleep_time = 2 ** attempt
                print(f"  Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"Failed after {max_retries} attempts: {e}")


# ==========================================================
# 🔹 MAIN FIRESTORE PROCESSOR
# ==========================================================
def process_firestore_queue(cred_path='jamaica-realtime-crisis-map-firebase-adminsdk-fbsvc-2b8d724ab7.json', project_id=None, collection_name='image_data'):
    """Fetch documents from Firestore, process any that lack classification, and update results."""
   
    records = fetch_firestore_image_records(collection_name)

    if not records:
        print(f'No documents found in Firestore collection "{collection_name}".')
        return

    print(f"\nFound {len(records)} documents in '{collection_name}'")
    processed, skipped, errors = 0, 0, 0

    for doc_id, rec in records.items():
        try:
            print(f"\n{'=' * 60}\nProcessing document: {doc_id}")

            # Skip already processed images (use is_classified heuristic)
            existing = rec.get('image_classification', {})
            if existing and is_classified(existing):
                print("  ⊘ Skipping: already classified")
                skipped += 1
                continue

            image_path = rec.get('image_path')
            metadata = rec.get('metadata', {})

            if not image_path:
                print("  ⊘ Skipping: no image_path field")
                skipped += 1
                continue

            signed_url = get_signed_url_from_storage_path(image_path)
            xy = rec.get('xy_coordinate', {})
            # Resolve coordinates from metadata or xy_coordinate (support dict/list)
            lat = metadata.get('latitude') or xy.get('latitude', None) if isinstance(xy, dict) else None
            lon = metadata.get('longitude') or xy.get('longitude', None) if isinstance(xy, dict) else None
            if lat is None or lon is None:
                # xy may be a list [lat, lon]
                if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                    try:
                        lat = lat or float(xy[0])
                        lon = lon or float(xy[1])
                    except Exception:
                        lat = lat or 0
                        lon = lon or 0
                else:
                    # Default to numeric 0 when nothing available
                    lat = lat or 0
                    lon = lon or 0

            # If coordinates are missing or zero, attempt to geocode from location description
            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except Exception:
                lat_f = 0
                lon_f = 0

            if (not lat_f or not lon_f):
                loc_desc = (metadata.get('location_description') or metadata.get('location_decription') or '').strip()
                if loc_desc:
                    geo = geocode_location(loc_desc)
                    if geo:
                        lat_f, lon_f = geo
                        # Write updated coordinates back to Firestore (metadata fields)
                        try:
                            client = firestore.client()
                            client.collection(collection_name).document(doc_id).update({
                                'metadata.latitude': lat_f,
                                'metadata.longitude': lon_f,
                                'metadata.xy_coordinate': [lat_f, lon_f]
                            })
                            print(f"  ✓ Geocoded '{loc_desc}' -> {lat_f},{lon_f} and updated document")
                        except Exception as e:
                            print(f"  ⚠️ Failed to update geocoded coords for {doc_id}: {e}")
                    else:
                        print(f"  ⚠️ Could not geocode location description: '{loc_desc}'")
            else:
                lat_f = lat_f
                lon_f = lon_f
            ts = metadata.get('timestamp', "")

            result = process_image_with_llm(signed_url, metadata, lat_f, lon_f, ts)
            update_firestore_classification(doc_id, result, collection_name)
            print(f"  ✓ Updated classification for {doc_id}")
            processed += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            errors += 1

    print(f"\n{'='*60}\nSummary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Total:     {len(records)}")


# ==========================================================
# 🔹 CLI ENTRYPOINT
# ==========================================================
def _run_cli():
    parser = argparse.ArgumentParser(description='Multimodal Vision Processing Helper')
    parser.add_argument('--run-firestore', action='store_true',
                        help='Process Firestore collection and update classifications')
    parser.add_argument('--service-account', default='serviceAccountKey.json',
                        help='Path to Firebase service account JSON')
    parser.add_argument('--project-id', default=FIREBASE_CONFIG.get('projectId'),
                        help='GCP project ID (optional)')
    parser.add_argument('--collection', default='image_data',
                        help='Firestore collection name')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode with a single image URL')
    args = parser.parse_args()

    if args.interactive:
        print("=" * 60)
        print("Interactive Vision Test")
        print("=" * 60)
        url = input("Image URL to analyze: ").strip()
        if not url:
            print("No URL provided. Exiting.")
            return
        meta = {"source": "local-test", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        result = process_image_with_llm(url, meta)
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("Firestore Vision Processing Batch Job")
        print("=" * 60)
        process_firestore_queue(args.service_account, args.project_id, args.collection)


if __name__ == "__main__":
    _run_cli()


def process_single_document(doc_id, collection_name='image_data'):
    """Process a single Firestore document by id: run the LLM on its image and update classification.

    Returns a tuple (success: bool, message: str).
    """
    client = firestore.client()
    doc_ref = client.collection(collection_name).document(doc_id)
    doc = doc_ref.get()
    if not doc.exists:
        return False, f"Document {doc_id} not found in collection {collection_name}"

    rec = doc.to_dict()
    try:
        image_path = rec.get('image_path') or rec.get('image_path', '')
        if not image_path:
            return False, f"Document {doc_id} has no image_path"

        signed_url = get_signed_url_from_storage_path(image_path)
        metadata = rec.get('metadata', {}) or {}

        # Resolve coordinates similar to batch processor
        xy = rec.get('xy_coordinate', {})
        lat = metadata.get('latitude') or (xy.get('latitude') if isinstance(xy, dict) else None)
        lon = metadata.get('longitude') or (xy.get('longitude') if isinstance(xy, dict) else None)
        if lat is None or lon is None:
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                try:
                    lat = float(xy[0])
                    lon = float(xy[1])
                except Exception:
                    lat = 0
                    lon = 0
            else:
                lat = lat or 0
                lon = lon or 0

        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            lat_f = 0
            lon_f = 0

        if (not lat_f or not lon_f):
            loc_desc = (metadata.get('location_description') or metadata.get('location_decription') or '')
            loc_desc = str(loc_desc).strip()
            if loc_desc:
                geo = geocode_location(loc_desc)
                if geo:
                    lat_f, lon_f = geo
                    try:
                        client.collection(collection_name).document(doc_id).update({
                            'metadata.latitude': lat_f,
                            'metadata.longitude': lon_f,
                            'metadata.xy_coordinate': [lat_f, lon_f]
                        })
                    except Exception as e:
                        # Non-fatal; continue
                        print(f"  ⚠️ Failed to update geocoded coords for {doc_id}: {e}")

        ts = metadata.get('timestamp', "")

        # Call LLM processing
        result = process_image_with_llm(signed_url, metadata, lat_f, lon_f, ts)

        # Validate result is a dict before writing
        if not isinstance(result, dict):
            return False, f"LLM returned non-dict result for {doc_id}: {type(result)}"

        update_firestore_classification(doc_id, result, collection_name)
        return True, f"Processed and updated classification for {doc_id}"

    except Exception as e:
        return False, f"Error processing {doc_id}: {e}"