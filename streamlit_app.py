"""
Streamlit app to visualize Jamaica Crisis Map data from Firestore.
Displays disaster response data with geocoded locations on an interactive map.
"""

import streamlit as st
import pandas as pd
import json
import time
import datetime

# Map and geocoding imports
import os
import subprocess
import tempfile
import uuid
import sys
import folium
from streamlit_folium import st_folium
from streamlit.components.v1 import html as st_html
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import firebase_admin
from firebase_admin import credentials, firestore


# Reuse Firebase initialization and helpers from the project's module
import multimodal_vision_processing as mmp

# Set page config to wide mode and custom title
st.set_page_config(
    page_title="Jamaica Crisis Map",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with collapsed sidebar for more map space
)

st.title("Jamaica Crisis Map — Disaster Response")

# Cached geocode helper (uses OpenStreetMap Nominatim)
@st.cache_data(ttl=86400)
def geocode_location(query: str):
    """Geocode a location string to (lat, lon) tuple."""
    if not query:
        return None
    try:
        geolocator = Nominatim(user_agent="crisis_map_app")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)
        loc = geocode(query)
        if loc:
            return (float(loc.latitude), float(loc.longitude))
    except Exception as e:
        st.warning(f"Geocoding error: {str(e)}")
        return None
    return None

@st.cache_data(ttl=30)
def load_records():
    """Load and process Firestore records for display."""
    records = mmp.fetch_firestore_image_records()
    rows = []
    for doc_id, data in records.items():
        lat = lon = 0
        location_description = ""
        meta = data.get('metadata', {})
        if meta:
            xy = meta.get('xy_coordinate', [0, 0])
            lat = xy[0] if isinstance(xy, list) and len(xy) >= 2 else 0
            lon = xy[1] if isinstance(xy, list) and len(xy) >= 2 else 0
        if lat == 0 and lon == 0:
            image_class = data.get('image_classification', {})
            if image_class and 'location' in image_class:
                loc = image_class['location']
                if loc.get('lat') != 0 or loc.get('lon') != 0:
                    lat = loc.get('lat', 0)
                    lon = loc.get('lon', 0)
                elif loc.get('location_description'):
                    location_description = loc['location_description']
                    coords = geocode_location(location_description)
                    if coords:
                        lat, lon = coords
                        try:
                            client = firestore.client()
                            client.collection('image_data').document(doc_id).update({
                                'image_classification.location.lat': lat,
                                'image_classification.location.lon': lon,
                                'metadata.latitude': lat,
                                'metadata.longitude': lon,
                                'metadata.xy_coordinate': [lat, lon]
                            })
                        except Exception as e:
                            st.warning(f"Failed to update coordinates for {doc_id}: {e}")
        timestamp = meta.get('timestamp') or data.get('image_classification', {}).get('timestamp', '')
        if isinstance(timestamp, (datetime.datetime, datetime.date)):
            timestamp = timestamp.isoformat()
        elif not isinstance(timestamp, str):
            timestamp = str(timestamp)
        hazards = data.get('image_classification', {}).get('hazards', [])
        health_risks = data.get('image_classification', {}).get('health_risks', [])
        logistics = data.get('image_classification', {}).get('logistics', [])
        # Ensure lists for dropdowns
        hazards_list = hazards if isinstance(hazards, list) else []
        health_risks_list = health_risks if isinstance(health_risks, list) else []
        logistics_list = logistics if isinstance(logistics, list) else []
        rows.append({
            'doc_id': doc_id,
            'lat': lat,
            'lon': lon,
            'timestamp': timestamp,
            'image_url': data.get('image_path', ''),
            'hazards': ', '.join(hazards) if hazards else '',
            'hazards_list': hazards_list,
            'health_risks': ', '.join(health_risks) if health_risks else '',
            'health_risks_list': health_risks_list,
            'logistics': ', '.join(logistics) if logistics else '',
            'logistics_list': logistics_list,
            'location_description': location_description,
            'raw': data
        })
    return pd.DataFrame(rows)

def create_map(df):
    """Create a Folium map with small circle markers for each record."""
    jamaica_center = [18.1096, -77.2975]
    m = folium.Map(
        location=jamaica_center,
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    for _, row in df.iterrows():
        if row['lat'] != 0 and row['lon'] != 0:
            popup_content = f"""
                <div style='width:250px'>
                    <p><strong>ID:</strong> {row['doc_id']}</p>
                    <p><strong>Location:</strong> {row['location_description']}</p>
                    <p><strong>Time:</strong> {row['timestamp']}</p>
                    {f"<p><strong>Hazards:</strong> {row['hazards']}</p>" if row['hazards'] else ''}
                    {f"<p><strong>Health Risks:</strong> {row['health_risks']}</p>" if row['health_risks'] else ''}
                    {f"<p><strong>Logistics:</strong> {row['logistics']}</p>" if row['logistics'] else ''}
                    {f"<img src='{row['image_url']}' style='max-width:100%;max-height:150px'>" if row['image_url'] else ''}
                </div>
            """
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=6,
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=row['location_description'] or row['doc_id'],
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                weight=2
            ).add_to(m)
    # Removed rectangle overlay (was showing an unwanted blue box around the map)
    return m

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    refresh = st.button("Refresh data from Firestore")
    search_term = st.text_input("Search records", "", help="Search in hazards, health risks, logistics, location descriptions, or doc id")
    st.markdown("#### Category Filters")
    hazards_filter = st.empty()
    health_risks_filter = st.empty()
    logistics_filter = st.empty()
    st.markdown("#### Map Bounds")
    min_lat = st.number_input("Min latitude", value=17.7, step=0.1, format="%.6f")
    max_lat = st.number_input("Max latitude", value=18.5, step=0.1, format="%.6f")
    min_lon = st.number_input("Min longitude", value=-78.4, step=0.1, format="%.6f")
    max_lon = st.number_input("Max longitude", value=-76.2, step=0.1, format="%.6f")
    show_only_unclassified = st.checkbox("Show only unclassified", value=False)

# --- Upload section: image or video ---
st.markdown("---")
st.header("Upload Media")
uploaded_file = st.file_uploader("Upload an image or a video (mp4, mov, jpg, png)", type=['mp4', 'mov', 'avi', 'mkv', 'jpg', 'jpeg', 'png'])

# Use session state to avoid duplicate uploads/creates across reruns
if 'last_uploaded_doc' not in st.session_state:
    st.session_state['last_uploaded_doc'] = None
if 'last_uploaded_tmp' not in st.session_state:
    st.session_state['last_uploaded_tmp'] = None

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_ext = os.path.splitext(file_name)[1].lower()
    st.info(f"Ready to upload: {file_name}")

    # Buttons: for videos, start preprocessing; for images, upload/create doc
    if file_ext in ('.mp4', '.mov', '.avi', '.mkv'):
        if st.button("Start preprocessing (extract frames & upload)"):
            with st.spinner("Saving uploaded video and running preprocessing..."):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                try:
                    tmp.write(uploaded_file.getbuffer())
                    tmp.flush()
                    tmp_path = tmp.name
                finally:
                    tmp.close()
                st.session_state['last_uploaded_tmp'] = tmp_path
                # Run preprocessing.py as a subprocess
                try:
                    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'preprocessing.py'), tmp_path]
                    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    st.text_area("Preprocessing output", proc.stdout + "\n" + proc.stderr, height=200)
                    st.success("Preprocessing finished. You can now run multimodal processing from the sidebar or below.")
                except subprocess.CalledProcessError as e:
                    st.error(f"Preprocessing failed: returncode={e.returncode}\n{e.stdout}\n{e.stderr}")
                finally:
                    # cleanup temp file
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                load_records.clear()
    else:
        if st.button("Upload & create Firestore document"):
            with st.spinner("Uploading image and creating document..."):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                try:
                    tmp.write(uploaded_file.getbuffer())
                    tmp.flush()
                    tmp_path = tmp.name
                finally:
                    tmp.close()
                try:
                    client = firestore.client()
                    bucket = mmp.storage.bucket()
                    dest_name = f"images/{uuid.uuid4().hex}_{os.path.basename(file_name)}"
                    blob = bucket.blob(dest_name)
                    blob.upload_from_filename(tmp_path)
                    try:
                        blob.make_public()
                        public_url = blob.public_url
                    except Exception:
                        public_url = mmp.get_signed_url_from_storage_path(f"gs://{bucket.name}/{dest_name}")
                    # Check if a document with the same image_path already exists to avoid duplicates
                    existing_docs = list(client.collection('image_data').where('image_path', '==', public_url).limit(1).stream())
                    if existing_docs:
                        existing_id = existing_docs[0].id
                        st.warning(f"A document with this image_path already exists: {existing_id}. Will not create a duplicate.")
                        st.session_state['last_uploaded_doc'] = existing_id
                    else:
                        doc_id = uuid.uuid4().hex
                        doc = {
                            'image_classification': {
                                'hazards': [],
                                'health_risks': [],
                                'logistics': [],
                                'people': [{'count': 0, 'activity': '', 'distress_signs': False}],
                                'location': {'lat': 0, 'lon': 0, 'location_description': ''},
                                'timestamp': ''
                            },
                            'image_path': public_url,
                            'metadata': {
                                'caption': '',
                                'source': file_name,
                                'timestamp': datetime.datetime.utcnow().isoformat(),
                                'xy_coordinate': [0, 0]
                            }
                        }
                        client.collection('image_data').document(doc_id).set(doc)
                        st.success(f"Uploaded and created document {doc_id}")
                        st.session_state['last_uploaded_doc'] = doc_id
                except Exception as e:
                    st.error(f"Failed to upload image and create document: {e}")
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                load_records.clear()

    # If a doc was just created in this session, show a button to run analysis on that single doc
    if st.session_state.get('last_uploaded_doc'):
        st.markdown("---")
        st.info(f"Most recent created doc: {st.session_state['last_uploaded_doc']}")
        if st.button("Run analysis for last uploaded document"):
            try:
                success, msg = mmp.process_single_document(st.session_state['last_uploaded_doc'])
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                # Refresh records after processing
                load_records.clear()
            except Exception as e:
                st.error(f"Error during multimodal processing: {e}")

# Load data
if refresh:
    load_records.clear()
    time.sleep(0.2)
df = load_records()

# Populate dropdowns for category filters
all_hazards = sorted({h for sublist in df['hazards_list'] for h in (sublist if isinstance(sublist, list) else [])})
all_health_risks = sorted({h for sublist in df['health_risks_list'] for h in (sublist if isinstance(sublist, list) else [])})
all_logistics = sorted({h for sublist in df['logistics_list'] for h in (sublist if isinstance(sublist, list) else [])})

with st.sidebar:
    selected_hazards = hazards_filter.multiselect("Hazards", options=all_hazards, default=[])
    selected_health_risks = health_risks_filter.multiselect("Health Risks", options=all_health_risks, default=[])
    selected_logistics = logistics_filter.multiselect("Logistics", options=all_logistics, default=[])

# Filter records
df_coords = df.dropna(subset=['lat', 'lon']).copy()
df_coords = df_coords[(df_coords['lat'] >= min_lat) & (df_coords['lat'] <= max_lat) &
                      (df_coords['lon'] >= min_lon) & (df_coords['lon'] <= max_lon)]

# Search filter
if search_term:
    search_lower = search_term.lower()
    mask = (
        df_coords['hazards'].str.lower().str.contains(search_lower, na=False) |
        df_coords['health_risks'].str.lower().str.contains(search_lower, na=False) |
        df_coords['logistics'].str.lower().str.contains(search_lower, na=False) |
        df_coords['location_description'].str.lower().str.contains(search_lower, na=False) |
        df_coords['doc_id'].str.lower().str.contains(search_lower, na=False)
    )
    df_coords = df_coords[mask]

# Category dropdown filters
if selected_hazards:
    df_coords = df_coords[df_coords['hazards_list'].apply(lambda x: any(h in x for h in selected_hazards))]
if selected_health_risks:
    df_coords = df_coords[df_coords['health_risks_list'].apply(lambda x: any(h in x for h in selected_health_risks))]
if selected_logistics:
    df_coords = df_coords[df_coords['logistics_list'].apply(lambda x: any(h in x for h in selected_logistics))]

if show_only_unclassified:
    df_coords = df_coords[~(df_coords['hazards'].str.len() > 0) & 
                         ~(df_coords['health_risks'].str.len() > 0) & 
                         ~(df_coords['logistics'].str.len() > 0)]

st.subheader(f"Map View — {len(df_coords)} records")
if search_term:
    st.caption(f"Showing results for search: '{search_term}'")

# Map height control (responsive-ish) - user can adjust height; width will fill the container
with st.sidebar:
    map_height = st.slider("Map height (px)", min_value=300, max_value=1200, value=600, step=50)

# Create and display map
if not df_coords.empty:
    m = create_map(df_coords)
    # Use direct HTML rendering to avoid streamlit_folium JSON-serialization issues
    # Use the folium HTML representation (works across folium versions)
    try:
        map_html = m._repr_html_()
    except Exception:
        # Fallback to get_root().render()
        map_html = m.get_root().render()

    # Render the HTML. height is controlled by the sidebar slider.
    st_html(map_html, height=map_height, scrolling=True)
else:
    st.info("No geolocated records found with current filters.")

# Details view
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Records")
    if df.empty:
        st.write("No documents found in collection 'image_data'")
    else:
        display_df = df[['doc_id', 'lat', 'lon', 'timestamp']].fillna('')
        st.dataframe(display_df.sort_values(by='timestamp', ascending=False))

with col2:
    st.subheader("Document Details")
    doc_ids = [''] + df['doc_id'].tolist()
    selected = st.selectbox('Select document', doc_ids)
    if selected:
        rec = df.set_index('doc_id').loc[selected]
        st.markdown(f"**Document ID:** {selected}")
        st.markdown(f"**Timestamp:** {rec['timestamp']}")
        if rec['location_description']:
            st.markdown(f"**Location:** {rec['location_description']}")
        
        st.markdown("**Image**")
        if rec['image_url']:
            try:
                st.image(rec['image_url'], width=350)
            except Exception:
                st.markdown(f"*Image URL (failed to load):* {rec['image_url']}")
        
        if rec['hazards'] or rec['health_risks'] or rec['logistics']:
            st.markdown("**Classifications**")
            if rec['hazards']: st.markdown(f"- **Hazards:** {rec['hazards']}")
            if rec['health_risks']: st.markdown(f"- **Health Risks:** {rec['health_risks']}")
            if rec['logistics']: st.markdown(f"- **Logistics:** {rec['logistics']}")
        
        with st.expander("Raw Document"):
            st.json(rec['raw'])

st.markdown("---")
st.caption("Data from Firestore collection 'image_data'. Refresh using sidebar button.")
