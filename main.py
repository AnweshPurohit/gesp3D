import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

# --- Configuration ---
WINDOW_NAME = "gesp3D - Particle Grid"
# Lower resolution for higher FPS/Lower Latency
WIDTH, HEIGHT = 854, 480 
# ADAPTIVE_ALPHA parameters
MIN_ALPHA = 0.02   # Very smooth (rock solid)
MAX_ALPHA = 0.3    # Controlled speed (prevents overshoot/vibration)
ROTATION_ALPHA = 0.05 # Aggressive smoothing for rotation to kill spinning
SPEED_THRESHOLD = 50.0 
NOISE_GATE = 6.0   # Slightly higher noise gate
MODEL_PATH = 'hand_landmarker.task'

def get_adaptive_alpha(dist_px):
    """Calculates smoothed alpha based on movement distance."""
    # Noise Gate: Ignore micro-jitters completely
    if dist_px < NOISE_GATE:
        return 0.0 # True deadstop for stability
        
    # Non-linear curve: fast ramp up
    # If moving, immediately jump to at least 0.1
    # Normalized speed 0..1
    norm_speed = min(1.0, (dist_px - NOISE_GATE) / (SPEED_THRESHOLD - NOISE_GATE))
    return MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * (norm_speed**0.5)

# --- Classes ---

class HandTracker:
    def __init__(self, model_path=MODEL_PATH):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5, 
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.landmarker = HandLandmarker.create_from_options(options)
        self.prev_landmarks = {} 
        self.start_time = time.time()
        
        # ID Tracking
        self.tracks = {} # ID -> Last Center (x, y)
        self.next_id = 0

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        hands_data = []
        h, w, c = img.shape
        
        # 1. Extract current frame hands
        if detection_result.hand_landmarks:
            for idx, hand_lms_list in enumerate(detection_result.hand_landmarks):
                # Convert to px
                current_lms_px = []
                for lm in hand_lms_list:
                    current_lms_px.append([int(lm.x * w), int(lm.y * h), lm.z * w])
                
                # wrist center
                wrist = current_lms_px[0]
                center = np.array(wrist[:2])
                
                hands_data.append({
                    "landmarks": current_lms_px,     # Will be smoothed
                    "raw_landmarks": list(current_lms_px), # Copy for gestures
                    "center": center,
                    "id": None
                })
        
        # 2. Match to existing tracks (Simple Euclidean Tracking)
        if len(self.tracks) > 0 and len(hands_data) > 0:
            track_ids = list(self.tracks.keys())
            
            # Simple greedy matching
            # Find closest pairs
            matches = []
            
            for h_idx, h_data in enumerate(hands_data):
                best_dist = 500 # Max jump
                best_tid = None
                
                for t_id in track_ids:
                    dist = np.linalg.norm(self.tracks[t_id] - h_data["center"])
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = t_id
                
                if best_tid is not None:
                    matches.append((best_dist, h_idx, best_tid))
            
            # Sort by distance (best matches first)
            matches.sort(key=lambda x: x[0])
            
            used_hands = set()
            used_tracks = set()
            
            for dist, h_idx, t_id in matches:
                if h_idx not in used_hands and t_id not in used_tracks:
                    hands_data[h_idx]["id"] = t_id
                    used_hands.add(h_idx)
                    used_tracks.add(t_id)
                    # Update track
                    self.tracks[t_id] = hands_data[h_idx]["center"]

        # 3. Assign new IDs
        track_keys = set(self.tracks.keys())
        
        for h_data in hands_data:
            if h_data["id"] is None:
                # Reuse 0/1 if free
                if 0 not in track_keys:
                    new_id = 0
                elif 1 not in track_keys:
                    new_id = 1
                else:
                    new_id = self.next_id
                    self.next_id += 1
                
                h_data["id"] = new_id
                self.tracks[new_id] = h_data["center"]
                track_keys.add(new_id)

        # 4. Prune lost tracks
        active_ids = {h["id"] for h in hands_data}
        self.tracks = {k: v for k, v in self.tracks.items() if k in active_ids}
        
        # 5. Smoothing
        output_hands = []
        for h_data in hands_data:
            t_id = h_data["id"]
            key = f"ID_{t_id}"
            current_lms_px = h_data["landmarks"]
            
            if key not in self.prev_landmarks:
                self.prev_landmarks[key] = current_lms_px
            else:
                prev_lms = self.prev_landmarks[key]
                prev_center = prev_lms[0][:2]
                curr_center = current_lms_px[0][:2]
                
                dist_moved = math.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
                alpha = get_adaptive_alpha(dist_moved)
                
                smoothed_lms = []
                for i in range(len(current_lms_px)):
                    curr = current_lms_px[i]
                    prev = prev_lms[i]
                    
                    new_x = alpha * curr[0] + (1 - alpha) * prev[0]
                    new_y = alpha * curr[1] + (1 - alpha) * prev[1]
                    new_z = alpha * curr[2] + (1 - alpha) * prev[2]
                    smoothed_lms.append([new_x, new_y, new_z])
                
                self.prev_landmarks[key] = smoothed_lms
                h_data["landmarks"] = smoothed_lms
            
            output_hands.append(h_data)

        # Sort by ID
        output_hands.sort(key=lambda x: x["id"])
        return output_hands

def is_pinched(landmarks, threshold=40):
    thumb = landmarks[4]
    index = landmarks[8]
    dist = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
    return dist < threshold

class ParticleCube:
    def __init__(self, size=200):
        self.size = size
        self.cached_points = None
        self.cached_divs = -1

    def generate_grid_points(self, divisions, points_per_line=20):
        div = max(1, int(divisions))
        step = self.size / div
        offset = self.size / 2.0
        
        points = []
        line_range = np.linspace(-offset, offset, points_per_line)
        
        for i in range(div + 1):
            for j in range(div + 1):
                fixed_y = -offset + i * step
                fixed_z = -offset + j * step
                p = np.zeros((len(line_range), 3))
                p[:, 0] = line_range
                p[:, 1] = fixed_y
                p[:, 2] = fixed_z
                points.append(p)
                
                fixed_x = -offset + i * step
                fixed_z = -offset + j * step
                p = np.zeros((len(line_range), 3))
                p[:, 0] = fixed_x
                p[:, 1] = line_range
                p[:, 2] = fixed_z
                points.append(p)

                fixed_x = -offset + i * step
                fixed_y = -offset + j * step
                p = np.zeros((len(line_range), 3))
                p[:, 0] = fixed_x
                p[:, 1] = fixed_y
                p[:, 2] = line_range
                points.append(p)
        
        return np.vstack(points)

    def project_points(self, center_pos, scale, rotation_matrix, divisions=1, view_distance=1000):
        idx_div = int(round(divisions))
        
        if self.cached_points is None or self.cached_divs != idx_div:
            density = 15 
            self.cached_points = self.generate_grid_points(idx_div, density)
            self.cached_divs = idx_div

        points_3d = self.cached_points
        rotated = points_3d @ rotation_matrix.T
        rotated *= scale
        z_coords = rotated[:, 2]
        
        depth_denom = view_distance + z_coords
        depth_denom[depth_denom == 0] = 0.0001
        
        factors = view_distance / depth_denom
        x_2d = rotated[:, 0] * factors + center_pos[0]
        y_2d = rotated[:, 1] * factors + center_pos[1]
        
        return np.column_stack((x_2d, y_2d, z_coords))

def get_rotation_matrix(pitch, yaw, roll):
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cz, sz = np.cos(roll),  np.sin(roll)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

# --- Main ---

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    
    if not cap.isOpened():
        print("Error: Webcam not found.")
        return

    try:
        detector = HandTracker()
    except Exception as e:
        print(f"Failed to load HandTracker: {e}")
        return

    cube = ParticleCube(size=250)
    
    print("Starting gesp3D Particle Engine...")
    print("Controls: Spread hands to zoom. Tilt hands for roll. Raise/Lower hands for pitch. Pinch (Both Hands) to subdivide.")
    
    curr_cx, curr_cy = WIDTH / 2.0, HEIGHT / 2.0
    curr_scale = 1.0
    curr_roll, curr_pitch, curr_yaw = 0.0, 0.0, 0.0
    curr_divs = 1.0 # Default to simple cube
    
    # Stretching state
    base_dist = 1.0
    base_scale = 1.0
    holding = False
    
    prev_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        black_bg = np.zeros_like(img)
        
        hands = detector.find_hands(img)
        
        target_pitch = curr_pitch * 0.95
        target_yaw = curr_yaw * 0.95
        target_roll = curr_roll 
        
        target_cx, target_cy = curr_cx, curr_cy
        target_scale = curr_scale
        target_divs = 1.0

        if len(hands) >= 2:
            # 1. Sort hands by X coordinate to ensure consistent Left/Right mapping
            # This prevents 180-degree jumps in atan2 if IDs swap.
            hands.sort(key=lambda x: x["center"][0])
            h_left = hands[0]
            h_right = hands[1]
            
            lms1 = h_left["landmarks"]
            lms2 = h_right["landmarks"]
            
            h1 = lms1[8] # ID 0 Tip (Smoothed)
            h2 = lms2[8] # ID 1 Tip (Smoothed)
            
            # Use RAW landmarks for pinch detection (Instant Response)
            pinched1 = is_pinched(h_left["raw_landmarks"])
            pinched2 = is_pinched(h_right["raw_landmarks"])
            
            # Control Logic
            target_cx = (h1[0] + h2[0]) / 2.0
            target_cy = (h1[1] + h2[1]) / 2.0
            
            # Roll: Angle between Hand 0 and Hand 1
            dy = h2[1] - h1[1]
            dx = h2[0] - h1[0]
            target_roll = math.atan2(dy, dx)
            
            # Stretching Logic
            current_dist = math.hypot(h2[0] - h1[0], h2[1] - h1[1])
            if (pinched1 or pinched2):
                if not holding:
                    holding = True
                    base_dist = max(current_dist, 10) # Avoid div by zero
                    base_scale = curr_scale
                
                target_scale = (current_dist / base_dist) * base_scale
            else:
                holding = False
                target_scale = curr_scale # Keep current scale if not pinched
            
            # Yaw: Relative Z
            dz = (h1[2] - h2[2]) 
            target_yaw = dz * 0.05 
            
            # Pitch: Avg Y
            avg_y = (h1[1] + h2[1]) / 2.0
            target_pitch = (avg_y - HEIGHT/2.0) * 0.005
            
            # Deadzones
            ROTATION_DEADZONE = 0.1 # Increased deadzone
            
            if abs(target_roll) < ROTATION_DEADZONE: target_roll = 0.0
            if abs(target_yaw) < ROTATION_DEADZONE: target_yaw = 0.0
            if abs(target_pitch) < ROTATION_DEADZONE: target_pitch = 0.0 
            
            if pinched1 and pinched2:
                raw_divs = max(1, (current_dist - 100) / 40.0) 
                target_divs = min(6, raw_divs) 
            else:
                target_divs = 1.0 
            
            # Indicators
            cv2.line(black_bg, (int(h1[0]), int(h1[1])), (int(h2[0]), int(h2[1])), (50, 50, 50), 1)
            c1 = (0, 255, 0) if pinched1 else (0, 0, 150)
            c2 = (0, 255, 0) if pinched2 else (0, 0, 150)
            cv2.circle(black_bg, (int(h1[0]), int(h1[1])), 8, c1, -1)
            cv2.circle(black_bg, (int(h2[0]), int(h2[1])), 8, c2, -1)
            
            if holding:
                status_text = "HOLDING & STRETCHING" if (pinched1 and not pinched2) or (not pinched1 and pinched2) else "SUBDIVIDING" if (pinched1 and pinched2) else "HOLDING"
                # Minimal, gentle status text at the bottom
                cv2.putText(black_bg, status_text, (WIDTH//2 - 80, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            
            # Smaller, more subtle ID tags
            cv2.putText(black_bg, f"ID {h_left['id']}", (int(h1[0]), int(h1[1]-15)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(black_bg, f"ID {h_right['id']}", (int(h2[0]), int(h2[1]-15)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Smooth Update - Adaptive
        pos_dist = math.hypot(target_cx - curr_cx, target_cy - curr_cy)
        pos_alpha = get_adaptive_alpha(pos_dist)
        
        curr_cx = pos_alpha * target_cx + (1 - pos_alpha) * curr_cx
        curr_cy = pos_alpha * target_cy + (1 - pos_alpha) * curr_cy
        curr_scale = pos_alpha * target_scale + (1 - pos_alpha) * curr_scale
        
        rot_diff = abs(target_roll - curr_roll) + abs(target_yaw - curr_yaw) + abs(target_pitch - curr_pitch)
        # Apply Angle Wrap logic for Roll to prevent 360-degree spinning
        diff_roll = target_roll - curr_roll
        if diff_roll > math.pi: diff_roll -= 2 * math.pi
        if diff_roll < -math.pi: diff_roll += 2 * math.pi
        
        curr_roll += ROTATION_ALPHA * diff_roll
        curr_yaw = ROTATION_ALPHA * target_yaw + (1 - ROTATION_ALPHA) * curr_yaw
        curr_pitch = ROTATION_ALPHA * target_pitch + (1 - ROTATION_ALPHA) * curr_pitch
        
        curr_divs = 0.1 * target_divs + 0.9 * curr_divs
        
        # --- Rendering ---
        center_pos = (curr_cx, curr_cy)
        rot_matrix = get_rotation_matrix(curr_pitch, curr_yaw, curr_roll)
        
        projected = cube.project_points(center_pos, curr_scale, rot_matrix, divisions=curr_divs)
        
        if len(projected) > 0:
            xs = projected[:, 0].astype(int)
            ys = projected[:, 1].astype(int)
            zs = projected[:, 2]
            
            h, w = black_bg.shape[:2]
            mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            
            v_xs = xs[mask]
            v_ys = ys[mask]
            v_zs = zs[mask]
            
            if len(v_xs) > 0:
                z_min, z_max = v_zs.min(), v_zs.max()
                if z_max == z_min: z_max += 1
                bins = np.linspace(z_max, z_min, 12) 
                
                for i in range(len(bins) - 1):
                    z_far = bins[i]
                    z_near = bins[i+1]
                    layer_mask = (v_zs <= z_far) & (v_zs > z_near)
                    
                    if not np.any(layer_mask): continue
                        
                    l_xs = v_xs[layer_mask]
                    l_ys = v_ys[layer_mask]
                    avg_z = (z_far + z_near) / 2
                    
                    val = int(max(0, min(255, 128 + avg_z * 0.5)))
                    if holding:
                        # Vibrant Cyan-Yellow shift when holding
                        color = (val, 255, 255 - val//2) 
                    else:
                        color = (255, val, 255 - val)
                    
                    if avg_z < -150:
                        for bx, by in zip(l_xs, l_ys): cv2.circle(black_bg, (bx, by), 3, color, -1)
                    elif avg_z < 0:
                        for bx, by in zip(l_xs, l_ys): cv2.circle(black_bg, (bx, by), 2, color, -1)
                    else:
                        black_bg[l_ys, l_xs] = color

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        # Subtle FPS counter
        cv2.putText(black_bg, f"FPS: {int(fps)}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        cv2.imshow(WINDOW_NAME, black_bg)
        
        if cv2.waitKey(1) & 0xFF == 27: 
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
