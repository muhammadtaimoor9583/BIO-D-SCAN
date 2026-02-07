import sys
import time
import argparse
import cv2
import numpy as np
from datetime import datetime
from picamera2 import MappedArray, Picamera2, Preview

# Import Hailo helper
from picamera2.devices import Hailo

# Import our custom modules
from src.tracker import InsectTracker
from src.database import save_to_cloud

# Global variables for the drawing callback (Live View)
current_tracks = {}

def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    detections = []
    for class_id, class_dets in enumerate(hailo_output):
        for det in class_dets:
            score = det[4]
            if score >= threshold:
                y0, x0, y1, x1 = det[:4]
                x1_px, y1_px = int(x0 * w), int(y0 * h)
                x2_px, y2_px = int(x1 * w), int(y1 * h)
                detections.append([x1_px, y1_px, x2_px, y2_px, score, class_names[class_id]])
    return detections

def draw_live_feed(request):
    """Draws active boxes on the live screen"""
    global current_tracks
    if current_tracks:
        with MappedArray(request, "main") as m:
            for t_id, track in current_tracks.items():
                x1, y1, x2, y2 = map(int, track['bbox'])
                
                # Draw Box
                cv2.rectangle(m.array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw ID
                label = f"ID:{t_id}"
                cv2.putText(m.array, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                # Optional: Draw faint trail on live view
                if len(track['history']) > 1:
                    pts = np.array(track['history'], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(m.array, [pts], False, (0, 0, 255), 1)

def main():
    global current_tracks
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="models/yolov8n_insects.hef")
    parser.add_argument("-l", "--labels", default="labels.txt")
    args = parser.parse_args()

    # Load Labels
    try:
        with open(args.labels, 'r') as f: labels = f.read().splitlines()
    except: return

    # Initialize Tracker
    # max_lost=20: If insect is gone for ~1 sec, we upload the track
    tracker = InsectTracker(max_lost=20)

    # Camera Setup
    video_w, video_h = 1280, 960
    
    with Hailo(args.model) as hailo:
        model_h, model_w, _ = hailo.get_input_shape()
        
        with Picamera2() as picam2:
            main_config = {'size': (video_w, video_h), 'format': 'XRGB8888'}
            lores_config = {'size': (model_w, model_h), 'format': 'RGB888'}
            
            # Reverted to simple controls (Auto exposure)
            controls = {'FrameRate': 30, 'AfMode': 2, 'AfRange': 0}
            
            config = picam2.create_preview_configuration(main_config, lores=lores_config, controls=controls)
            picam2.configure(config)
            picam2.start_preview(Preview.QTGL, x=0, y=0, width=video_w, height=video_h)
            picam2.start()
            picam2.pre_callback = draw_live_feed

            print("🐞 Tracker Started! Waiting for insects to leave the frame...")

            try:
                while True:
                    # 1. Capture Frames
                    frame = picam2.capture_array('lores')
                    
                    # We capture the high-res frame here to pass to the tracker
                    # This ensures we save the "Cleanest" image for the report
                    high_res_frame = picam2.capture_array('main')
                    
                    # 2. Inference
                    results = hailo.run(frame)
                    formatted_dets = extract_detections(results, video_w, video_h, labels)

                    # 3. Update Tracker
                    # RETURNS: active_tracks (dict), finished_tracks (list)
                    current_tracks, finished_tracks = tracker.update(formatted_dets, high_res_frame)

                    # 4. Handle Insects that just LEFT
                    for track in finished_tracks:
                        t_id = track['id']
                        print(f"Insect {t_id} finished track. Uploading...")
                        
                        # Get the best image
                        final_img = track['best_image']
                        
                        # Picamera gives XRGB, Convert to BGR for OpenCV saving
                        final_img = cv2.cvtColor(final_img, cv2.COLOR_RGBA2BGR)
                        
                        # --- DRAW THE PATH LINE ---
                        if len(track['history']) > 1:
                            pts = np.array(track['history'], np.int32).reshape((-1, 1, 2))
                            
                            # Draw Yellow Line (Thickness 2)
                            cv2.polylines(final_img, [pts], False, (0, 255, 255), 2)
                            
                            # Draw Start (Red Dot) and End (Green Dot)
                            cv2.circle(final_img, track['history'][0], 4, (0, 0, 255), -1)
                            cv2.circle(final_img, track['history'][-1], 6, (0, 255, 0), -1)
                        
                        # Save Locally
                        temp_filename = f"track_{t_id}.jpg"
                        cv2.imwrite(temp_filename, final_img)
                        
                        # Prepare Data (serialize datetime to ISO string for JSON compatibility)
                        data_payload = {
                            'id': t_id,
                            'type': track['type'],
                            'timestamp': datetime.now().isoformat(),
                            'confidence': float(track['best_score']),
                            # Enhanced tracking data
                            'path_points': track.get('path_points', []),
                            'distance_traveled': track.get('distance_traveled'),
                            'duration_seconds': track.get('duration_seconds'),
                            'entry_point': track.get('entry_point'),
                            'exit_point': track.get('exit_point'),
                            'frame_count': track.get('frame_count')
                        }
                        
                        # Upload
                        save_to_cloud(data_payload, temp_filename)

            except KeyboardInterrupt:
                picam2.stop()

if __name__ == "__main__":
    main()