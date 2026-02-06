import time
import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment

class InsectTracker:
    def __init__(self, max_lost=20, min_distance=15, 
                 distance_weight=0.6, area_weight=0.4, cost_threshold=0.8,
                 frame_width=1280, frame_height=960):
        self.next_id = 0
        self.tracks = {} 
        self.max_lost = max_lost 
        self.min_distance = min_distance
        
        # Hungarian algorithm cost function weights
        self.distance_weight = distance_weight
        self.area_weight = area_weight
        self.cost_threshold = cost_threshold
        self.max_frame_distance = 2000  # Normalizer for frame diagonal
        
        # Frame dimensions for edge detection
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.edge_margin = 50  # Pixels from edge to consider as entry/exit

    def update(self, detections, current_frame_image):
        updated_active_ids = []
        finished_tracks = [] 
        
        # Get list of active tracks (not lost for too long)
        active_track_ids = [t_id for t_id, track in self.tracks.items() if track['lost'] <= 5]
        active_tracks = [self.tracks[t_id] for t_id in active_track_ids]
        
        # Skip matching if no tracks or no detections
        if len(active_tracks) > 0 and len(detections) > 0:
            # Build cost matrix using Hungarian algorithm approach
            cost_matrix = self.calc_cost_matrix(active_tracks, detections)
            
            # Optimal assignment using Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Process matched pairs
            matched_det_indices = set()
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < self.cost_threshold:
                    # Valid match - update track
                    t_id = active_track_ids[i]
                    det = detections[j]
                    x1, y1, x2, y2, score, cls = det
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    
                    self.tracks[t_id]['bbox'] = [x1, y1, x2, y2]
                    self.tracks[t_id]['history'].append(centroid)
                    self.tracks[t_id]['lost'] = 0
                    self.tracks[t_id]['class_votes'].append(cls)
                    self.tracks[t_id]['frame_count'] += 1
                    updated_active_ids.append(t_id)
                    
                    # Save best image (highest confidence)
                    if score > self.tracks[t_id]['best_score']:
                        self.tracks[t_id]['best_score'] = score
                        self.tracks[t_id]['best_image'] = current_frame_image.copy()
                    
                    matched_det_indices.add(j)
            
            # Create new tracks for unmatched detections
            for j, det in enumerate(detections):
                if j not in matched_det_indices:
                    self._create_new_track(det, current_frame_image)
                    updated_active_ids.append(self.next_id - 1)
        else:
            # No active tracks - create new tracks for all detections
            for det in detections:
                self._create_new_track(det, current_frame_image)
                updated_active_ids.append(self.next_id - 1)
                
        # Process Lost Tracks
        track_ids = list(self.tracks.keys())
        for t_id in track_ids:
            if t_id not in updated_active_ids:
                self.tracks[t_id]['lost'] += 1
                
                if self.tracks[t_id]['lost'] > self.max_lost:
                    track = self.tracks[t_id]
                    
                    # Filter by Distance (Did it actually move?)
                    start_pt = np.array(track['history'][0])
                    end_pt = np.array(track['history'][-1])
                    dist = np.linalg.norm(end_pt - start_pt)
                    
                    if dist > self.min_distance:
                        # Finalize track data
                        self._finalize_track(track, dist)
                        finished_tracks.append(track)
                    else:
                        print(f"🗑️ Discarded ID {t_id} (Moved only {int(dist)}px)")

                    del self.tracks[t_id]
                    
        return self.tracks, finished_tracks

    def _create_new_track(self, det, current_frame_image):
        """Create a new track for an unmatched detection."""
        x1, y1, x2, y2, score, cls = det
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        self.tracks[self.next_id] = {
            'id': self.next_id,
            'bbox': [x1, y1, x2, y2],
            'history': [centroid],
            'lost': 0,
            'class_votes': [cls],
            'best_score': score,
            'best_image': current_frame_image.copy(),
            # NEW: Timing and edge data
            'start_time': time.time(),
            'entry_point': self._get_edge_name(centroid),
            'frame_count': 1
        }
        self.next_id += 1

    def _finalize_track(self, track, distance):
        """Finalize track with computed statistics."""
        # Resolve Class (Democracy Vote)
        vote_counts = Counter(track['class_votes'])
        most_common_class, count = vote_counts.most_common(1)[0]
        
        track['type'] = most_common_class
        track['final_confidence'] = count / len(track['class_votes'])
        track['distance_traveled'] = float(distance)
        track['duration_seconds'] = time.time() - track['start_time']
        track['exit_point'] = self._get_edge_name(track['history'][-1])
        
        # Convert history to serializable format
        track['path_points'] = [[int(x), int(y)] for x, y in track['history']]

    def _get_edge_name(self, point):
        """Determine which edge of the frame a point is closest to."""
        x, y = point
        
        distances = {
            'top': y,
            'bottom': self.frame_height - y,
            'left': x,
            'right': self.frame_width - x
        }
        
        closest_edge = min(distances, key=distances.get)
        
        # Only return edge name if within margin
        if distances[closest_edge] <= self.edge_margin:
            return closest_edge
        return 'center'

    def calc_cost_matrix(self, tracks, detections):
        """Calculate cost matrix for Hungarian algorithm assignment."""
        n_tracks = len(tracks)
        n_dets = len(detections)
        cost_matrix = np.full((n_tracks, n_dets), 1e6)
        
        for i, track in enumerate(tracks):
            track_bbox = track['bbox']
            track_cx = (track_bbox[0] + track_bbox[2]) / 2
            track_cy = (track_bbox[1] + track_bbox[3]) / 2
            track_w = track_bbox[2] - track_bbox[0]
            track_h = track_bbox[3] - track_bbox[1]
            
            for j, det in enumerate(detections):
                x1, y1, x2, y2, score, cls = det
                det_cx = (x1 + x2) / 2
                det_cy = (y1 + y2) / 2
                det_w = x2 - x1
                det_h = y2 - y1
                
                dist = self.calc_euclidean_distance(track_cx, track_cy, det_cx, det_cy)
                norm_dist = dist / self.max_frame_distance
                
                area_ratio = self.calc_area_ratio(track_w, track_h, det_w, det_h)
                area_dissimilarity = 1 - area_ratio
                
                cost = (norm_dist * self.distance_weight) + (area_dissimilarity * self.area_weight)
                cost_matrix[i, j] = cost
        
        return cost_matrix

    def calc_euclidean_distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance between two points."""
        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    
    def calc_area_ratio(self, w1, h1, w2, h2):
        """Calculate area ratio between two bounding boxes."""
        area1 = w1 * h1
        area2 = w2 * h2
        if area1 == 0 or area2 == 0:
            return 0
        return min(area1, area2) / max(area1, area2)

    def calculate_iou(self, boxA, boxB):
        """Calculate Intersection over Union (kept for compatibility)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        denom = float((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) + 
                      (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) - interArea)
        return interArea / denom if denom > 0 else 0
    
    def get_session_stats(self):
        """Get statistics for current tracking session."""
        return {
            'total_tracks_created': self.next_id,
            'active_tracks': len(self.tracks)
        }
