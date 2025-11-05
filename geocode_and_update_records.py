"""
geocode_and_update_records.py

Scan Firestore `image_data` documents and, when a document has a
`metadata.location_description` (or misspelled `location_decription`) but missing
latitude/longitude, geocode the description and update the Firestore document
with the resulting coordinates.

Usage:
    python geocode_and_update_records.py --dry-run

Notes:
- Uses Nominatim (OpenStreetMap) via geopy. Rate-limited and cached in-memory.
- Be polite to Nominatim: the script default rate is 1 request / second.
- Make sure your Firebase service account JSON is available and multimodal_vision_processing
  initializes Firebase as expected (the script imports that module to reuse init).
"""

import time
import json
import argparse
from typing import Optional, Tuple

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Reuse Firebase initialization from project module
import multimodal_vision_processing as mmp
from firebase_admin import firestore

# Simple in-memory cache to avoid repeated geocoding calls during a run
GEOCODE_CACHE = {}


def geocode_location(query: str, delay: float = 1.0) -> Optional[Tuple[float, float]]:
    """Geocode a query string using Nominatim. Returns (lat, lon) or None."""
    if not query:
        return None
    q = query.strip().lower()
    if q in GEOCODE_CACHE:
        return GEOCODE_CACHE[q]

    geolocator = Nominatim(user_agent="crisis_map_geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=delay, max_retries=2)
    try:
        loc = geocode(query)
        if loc:
            result = (float(loc.latitude), float(loc.longitude))
            GEOCODE_CACHE[q] = result
            return result
    except Exception as e:
        # Don't raise; caller will handle and log
        print(f"  geocode error for '{query}': {e}")
        return None
    return None


def needs_geocode(rec: dict) -> bool:
    """Return True if the record lacks useful coordinates and has a location description."""
    metadata = rec.get('metadata', {}) or {}
    # Check existing coordinates
    lat = metadata.get('latitude') or metadata.get('lat')
    lon = metadata.get('longitude') or metadata.get('lon')
    xy = metadata.get('xy_coordinate') or rec.get('xy_coordinate')

    def is_valid_num(v):
        try:
            v = float(v)
            return abs(v) > 1e-6
        except Exception:
            return False

    if is_valid_num(lat) and is_valid_num(lon):
        return False
    if isinstance(xy, (list, tuple)) and len(xy) >= 2 and is_valid_num(xy[0]) and is_valid_num(xy[1]):
        return False

    # Check for location_description (and common misspelling)
    location_description = metadata.get('location_description') or metadata.get('location_decription')
    return bool(location_description and str(location_description).strip())


def update_record_with_coords(doc_id: str, collection: str, lat: float, lon: float, dry_run: bool = True):
    """Update Firestore document metadata with coordinates. Uses nested update fields."""
    db = firestore.client()
    doc_ref = db.collection(collection).document(doc_id)

    # Update both metadata.latitude/longitude and metadata.xy_coordinate for compatibility
    updates = {
        'metadata.latitude': lat,
        'metadata.longitude': lon,
        'metadata.xy_coordinate': [lat, lon]
    }

    if dry_run:
        print(f"  [dry-run] would update {doc_id} -> {updates}")
        return True

    try:
        doc_ref.update(updates)
        print(f"  Updated {doc_id} with lat={lat}, lon={lon}")
        return True
    except Exception as e:
        print(f"  Failed to update {doc_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Geocode records with location_description and update Firestore')
    parser.add_argument('--collection', default='image_data', help='Firestore collection name')
    parser.add_argument('--dry-run', action='store_true', help='Do not write updates to Firestore')
    parser.add_argument('--limit', type=int, help='Limit number of documents to process')
    parser.add_argument('--delay', type=float, default=1.0, help='Seconds delay between geocode requests')
    parser.add_argument('--cache-file', help='Optional path to save/load geocode cache (JSON)')
    args = parser.parse_args()

    if args.cache_file:
        try:
            with open(args.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for k, v in data.items():
                    GEOCODE_CACHE[k] = tuple(v)
            print(f"Loaded {len(GEOCODE_CACHE)} cached geocodes from {args.cache_file}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: failed to load cache file: {e}")

    print("Scanning Firestore collection: ", args.collection)
    records = mmp.fetch_firestore_image_records(args.collection)

    total = 0
    geocoded = 0
    skipped = 0
    failed = 0

    for doc_id, rec in records.items():
        if args.limit and total >= args.limit:
            break
        total += 1

        if not needs_geocode(rec):
            skipped += 1
            continue

        metadata = rec.get('metadata', {}) or {}
        location_description = metadata.get('location_description') or metadata.get('location_decription')
        if not location_description:
            skipped += 1
            continue

        print(f"Processing {doc_id}: '{location_description}'")
        geo = geocode_location(location_description, delay=args.delay)
        if geo:
            lat, lon = geo
            ok = update_record_with_coords(doc_id, args.collection, lat, lon, dry_run=args.dry_run)
            if ok:
                geocoded += 1
            else:
                failed += 1
        else:
            print(f"  Could not geocode '{location_description}'")
            failed += 1

    print("\nSummary:")
    print(f"  Total scanned: {total}")
    print(f"  Skipped (already had coords): {skipped}")
    print(f"  Geocoded & updated: {geocoded}")
    print(f"  Failed: {failed}")

    if args.cache_file:
        try:
            with open(args.cache_file, 'w', encoding='utf-8') as f:
                json.dump({k: list(v) for k, v in GEOCODE_CACHE.items()}, f, ensure_ascii=False, indent=2)
            print(f"Saved geocode cache to {args.cache_file}")
        except Exception as e:
            print(f"Warning: failed to save cache file: {e}")


if __name__ == '__main__':
    main()
