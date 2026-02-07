"""
Video-based testing script for Bio-D-Scan.
Use this to test the tracker and database without a camera.

Usage:
    python test_video.py --video insects.mp4 --model models/yolov8n.pt
    python test_video.py --video insects.mp4 --model models/yolov8n.pt --display
"""

import sys
import time
import argparse
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# Import our custom modules
from src.tracker import InsectTracker
from src.database import save_to_cloud, start_session, end_session

def main():
    parser = argparse.ArgumentParser(description="Test Bio-D-Scan with video file")
    parser.add_argument("-v", "--video", required=True, help="Path to input video file")
    parser.add_argument("-m", "--model", default="models/yolov8n.pt", help="Path to YOLO model (.pt or .onnx)")
    parser.add_argument("-l", "--labels", default="labels.txt", help="Path to labels file")
    parser.add_argument("-d", "--display", action="store_true", help="Show live preview window")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading to Supabase (local test only)")
    args = parser.parse_args()

    # Validate video path
    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        return

    # Load Labels
    try:
        with open(args.labels, 'r') as f:
            labels = f.read().splitlines()
        print(f"📋 Loaded {len(labels)} classes: {labels}")
    except FileNotFoundError:
        print(f"❌ Labels file not found: {args.labels}")
        return

    # Load YOLO model (using ultralytics)
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
        print(f"✅ Loaded model: {args.model}")
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        return
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Initialize Tracker
    tracker = InsectTracker(max_lost=20, frame_width=1280, frame_height=960)
    
    # Open Video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {args.video}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {video_w}x{video_h} @ {video_fps}fps, {total_frames} frames")

    # Start session if not skipping upload
    if not args.skip_upload:
        start_session(location="Video Test")

    print("🐞 Tracker Started! Processing video...")
    
    frame_count = 0
    total_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO inference
            results = model(frame, verbose=False, conf=args.conf)
            
            # Format detections for tracker
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = labels[cls_id] if cls_id < len(labels) else f"class_{cls_id}"
                        detections.append([x1, y1, x2, y2, conf, cls_name])
            
            # Update Tracker
            current_tracks, finished_tracks = tracker.update(detections, frame)
            
            # Handle finished tracks
            for track in finished_tracks:
                t_id = track['id']
                total_detections += 1
                print(f"✅ Track {t_id} finished: {track['type']} (conf: {track.get('final_confidence', 0):.2f})")
                
                if not args.skip_upload:
                    # Get the best image
                    final_img = track['best_image'].copy()
                    
                    # Draw the path line
                    if len(track['history']) > 1:
                        pts = np.array(track['history'], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(final_img, [pts], False, (0, 255, 255), 2)
                        cv2.circle(final_img, track['history'][0], 4, (0, 0, 255), -1)
                        cv2.circle(final_img, track['history'][-1], 6, (0, 255, 0), -1)
                    
                    # Save locally
                    temp_filename = f"track_{t_id}.jpg"
                    cv2.imwrite(temp_filename, final_img)
                    
                    # Prepare Data
                    data_payload = {
                        'id': t_id,
                        'type': track['type'],
                        'timestamp': datetime.now().isoformat(),
                        'confidence': float(track['best_score']),
                        'path_points': track.get('path_points', []),
                        'distance_traveled': track.get('distance_traveled'),
                        'duration_seconds': track.get('duration_seconds'),
                        'entry_point': track.get('entry_point'),
                        'exit_point': track.get('exit_point'),
                        'frame_count': track.get('frame_count')
                    }
                    
                    # Upload
                    save_to_cloud(data_payload, temp_filename)
            
            # Display if requested
            if args.display:
                display_frame = frame.copy()
                
                # Draw active tracks
                for t_id, track in current_tracks.items():
                    x1, y1, x2, y2 = map(int, track['bbox'])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"ID:{t_id}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw trail
                    if len(track['history']) > 1:
                        pts = np.array(track['history'], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(display_frame, [pts], False, (0, 0, 255), 1)
                
                # Add stats overlay
                cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Active: {len(current_tracks)} | Total: {total_detections}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Bio-D-Scan Test", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)  # Pause
            
            # Progress update every 100 frames
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📊 Progress: {progress:.1f}% ({frame_count}/{total_frames})")

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if not args.skip_upload:
            end_session()
        
        print(f"\n📊 Summary:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total detections: {total_detections}")
        print(f"   Active tracks at end: {len(current_tracks)}")

if __name__ == "__main__":
    main()
