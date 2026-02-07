import os
import json
import time
import shutil
import glob
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class QueueManager:
    """
    Manages a local queue for failed uploads (Store & Forward).
    """
    def __init__(self, queue_dir="queue"):
        self.queue_dir = queue_dir
        if not os.path.exists(self.queue_dir):
            os.makedirs(self.queue_dir)
            
    def enqueue(self, track_data, image_path):
        """
        Saves track data and image to the local queue.
        """
        try:
            timestamp = int(datetime.now().timestamp())
            track_id = track_data.get('id', 'unknown')
            base_name = f"{timestamp}_{track_id}"
            
            # 1. Move Image
            # We copy instead of move if we want to keep the original for debug, 
            # but for this app, moving is better to clean up temp files.
            ext = os.path.splitext(image_path)[1]
            queued_image_path = os.path.join(self.queue_dir, f"{base_name}{ext}")
            
            # If source file exists, move it. If not (weird race condition), skip.
            if os.path.exists(image_path):
                shutil.copy(image_path, queued_image_path)
            
            # 2. Save Metadata
            # Add the queue image path so we know which file belongs to this data
            track_data['queued_image_path'] = queued_image_path
            
            json_path = os.path.join(self.queue_dir, f"{base_name}.json")
            with open(json_path, 'w') as f:
                json.dump(track_data, f, cls=DateTimeEncoder)
                
            print(f"Internet down? Saved to queue: {base_name}")
            return True
        except Exception as e:
            print(f"Failed to queue data: {e}")
            return False

    def get_pending_items(self):
        """
        Returns a list of pending JSON files in the queue.
        """
        files = glob.glob(os.path.join(self.queue_dir, "*.json"))
        # Sort by oldest first
        files.sort()
        return files

    def remove_item(self, json_path):
        """
        Cleans up the JSON and its associated image after successful upload.
        """
        try:
            # Read JSON to find image path
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                image_path = data.get('queued_image_path')
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                
                os.remove(json_path)
        except Exception as e:
            print(f"Error cleaning up queue item {json_path}: {e}")

    def purge_queue(self):
        """
        DANGER: Deletes all items in queue.
        """
        if os.path.exists(self.queue_dir):
            shutil.rmtree(self.queue_dir)
        os.makedirs(self.queue_dir)
