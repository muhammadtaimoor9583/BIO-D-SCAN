import os
import time
import threading
import json
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from src.queue_manager import QueueManager

# --- CONFIGURATION ---
load_dotenv()

URL = os.environ.get("PROJECT_URL")
KEY = os.environ.get("API_KEY")

if not URL or not KEY:
    print("ERROR: PROJECT_URL or API_KEY not found in .env file!")

# Constants
BUCKET_NAME = "insects"
TABLE_NAME = "tracks"

# Initialize Supabase Client
try:
    supabase: Client = create_client(URL, KEY)
except Exception as e:
    print(f"Failed to initialize Supabase: {e}")
    supabase = None

# Initialize Queue
queue = QueueManager()

# --- SESSION MANAGEMENT ---
current_session_id = None
device_id = os.environ.get("DEVICE_ID", "rpi5_default")

def start_session(location=None):
    """Start a new monitoring session. Call this when the app starts."""
    global current_session_id
    
    if not supabase:
        print("⚠️ Supabase not available, session not created")
        return None
    
    try:
        result = supabase.table('sessions').insert({
            "device_id": device_id,
            "started_at": datetime.now().isoformat(),
            "location": location,
            "is_active": True,
            "total_detections": 0
        }).execute()
        
        current_session_id = result.data[0]['id']
        print(f"📍 Session started: {current_session_id}")
        return current_session_id
    except Exception as e:
        print(f"Failed to start session: {e}")
        return None

def end_session():
    """End the current monitoring session. Call this when the app stops."""
    global current_session_id
    
    if not supabase or not current_session_id:
        return
    
    try:
        supabase.table('sessions').update({
            "ended_at": datetime.now().isoformat(),
            "is_active": False
        }).eq('id', current_session_id).execute()
        
        print(f"📍 Session ended: {current_session_id}")
        current_session_id = None
    except Exception as e:
        print(f"Failed to end session: {e}")

def update_session_stats(insect_type):
    """Update statistics for the current session."""
    if not supabase or not current_session_id:
        return
    
    try:
        # Increment total detections
        supabase.rpc('increment_session_detections', {
            'session_id_param': current_session_id
        }).execute()
        
        # Upsert statistics
        supabase.table('statistics').upsert({
            "session_id": current_session_id,
            "insect_type": insect_type,
            "count": 1,  # Will be incremented by trigger
            "last_updated": datetime.now().isoformat()
        }, on_conflict='session_id,insect_type').execute()
    except Exception as e:
        # Statistics update is non-critical
        print(f"Stats update failed (non-critical): {e}")


def upload_to_supabase(track_data, image_path):
    """
    Upload track data and image to Supabase.
    Returns True if successful, False otherwise.
    """
    if not supabase:
        return False

    try:
        # Generate filename
        if 'queued_image_path' in track_data:
            file_name = os.path.basename(track_data['queued_image_path'])
        else:
            timestamp = track_data.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp_str = str(int(datetime.fromisoformat(timestamp).timestamp()))
            else:
                timestamp_str = str(int(timestamp.timestamp()))
            file_name = f"{track_data['id']}_{timestamp_str}.jpg"

        # 1. Upload Image
        with open(image_path, 'rb') as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=file_name,
                file=f,
                file_options={"content-type": "image/jpeg"}
            )
        
        # 2. Get Public URL
        image_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)

        # 3. Prepare enhanced payload
        timestamp = track_data.get('timestamp', datetime.now())
        if not isinstance(timestamp, str):
            timestamp = timestamp.isoformat()
        
        data_payload = {
            "tracker_id": track_data['id'],
            "type": track_data['type'],
            "confidence": track_data.get('confidence') or track_data.get('final_confidence', 0),
            "timestamp": timestamp,
            "image_url": image_url,
            # NEW: Enhanced tracking data
            "path_points": track_data.get('path_points', []),
            "distance_traveled": track_data.get('distance_traveled'),
            "duration_seconds": track_data.get('duration_seconds'),
            "entry_point": track_data.get('entry_point'),
            "exit_point": track_data.get('exit_point'),
            "frame_count": track_data.get('frame_count')
        }
        
        # Add session if available
        if current_session_id:
            data_payload["session_id"] = current_session_id
        
        # 4. Insert Data
        supabase.table(TABLE_NAME).insert(data_payload).execute()
        
        # 5. Update session statistics
        update_session_stats(track_data['type'])
        
        return True

    except Exception as e:
        print(f"Upload Error: {e}")
        return False


def flush_queue_thread():
    """Background thread that retries failed uploads every 60 seconds."""
    while True:
        pending_files = queue.get_pending_items()
        if pending_files:
            print(f"Attempting to flush {len(pending_files)} pending items...")
            
            for json_path in pending_files:
                try:
                    with open(json_path, 'r') as f:
                        track_data = json.load(f)
                    
                    image_path = track_data.get('queued_image_path')
                    
                    if image_path and os.path.exists(image_path):
                        success = upload_to_supabase(track_data, image_path)
                        if success:
                            print(f"✅ Flushed: {track_data.get('id')}")
                            queue.remove_item(json_path)
                        else:
                            break  # Internet probably still down
                    else:
                        print(f"Missing image for {json_path}, removing corrupted item.")
                        queue.remove_item(json_path)
                        
                except Exception as e:
                    print(f"Error flushing item: {e}")
        
        time.sleep(60)

# Start the flusher thread
flusher = threading.Thread(target=flush_queue_thread, daemon=True)
flusher.start()


def upload_track_thread(track_data, image_path):
    """Runs in background thread to upload. Queues on failure."""
    try:
        print(f"📤 Uploading Track {track_data['id']} ({track_data.get('type', 'unknown')})...")
        success = upload_to_supabase(track_data, image_path)
        
        if success:
            print(f"✅ Upload Complete: {track_data['id']}")
            if os.path.exists(image_path):
                os.remove(image_path)
        else:
            queue.enqueue(track_data, image_path)
            if os.path.exists(image_path):
                os.remove(image_path)

    except Exception as e:
        print(f"Critical Error in upload thread: {e}")
        queue.enqueue(track_data, image_path)


def save_to_cloud(track_data, image_path):
    """Non-blocking wrapper to start upload in a separate thread."""
    thread = threading.Thread(target=upload_track_thread, args=(track_data, image_path))
    thread.start()


def get_session_summary():
    """Get summary of current session for dashboard display."""
    if not supabase or not current_session_id:
        return None
    
    try:
        # Get session info
        session = supabase.table('sessions').select('*').eq('id', current_session_id).single().execute()
        
        # Get statistics for session
        stats = supabase.table('statistics').select('*').eq('session_id', current_session_id).execute()
        
        return {
            'session': session.data,
            'statistics': stats.data
        }
    except Exception as e:
        print(f"Failed to get session summary: {e}")
        return None