import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import tkinter as tk
from threading import Thread
import queue

# --- Configuration ---
WINDOW_NAME = "gesp3D - Sandbox"
WIDTH, HEIGHT = 854, 480 
MIN_ALPHA = 0.04   
MAX_ALPHA = 0.6    # Faster adaptation
ROTATION_ALPHA = 0.18 # Snappier rotation
SPEED_THRESHOLD = 30.0 
NOISE_GATE = 3.0   # More sensitive
MODEL_PATH = 'hand_landmarker.task'

def get_adaptive_alpha(dist_px):
    if dist_px < NOISE_GATE:
        return 0.0 
    norm_speed = min(1.0, (dist_px - NOISE_GATE) / (SPEED_THRESHOLD - NOISE_GATE))
    return MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * (norm_speed**0.6)

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
            min_hand_detection_confidence=0.45, # Lowered to 'catch' hands easier
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.6 # High for stability
        )
        
        self.landmarker = HandLandmarker.create_from_options(options)
        self.prev_landmarks = {} 
        self.start_time = time.time()
        self.tracks = {} 
        self.next_id = 0

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        hands_data = []
        h, w, c = img.shape
        
        if detection_result.hand_landmarks:
            for idx, hand_lms_list in enumerate(detection_result.hand_landmarks):
                current_lms_px = []
                for lm in hand_lms_list:
                    current_lms_px.append([int(lm.x * w), int(lm.y * h), lm.z * w])
                
                wrist = current_lms_px[0]
                center = np.array(wrist[:2])
                
                hands_data.append({
                    "landmarks": current_lms_px,     
                    "raw_landmarks": list(current_lms_px), 
                    "center": center,
                    "id": None
                })
        
        if len(self.tracks) > 0 and len(hands_data) > 0:
            track_ids = list(self.tracks.keys())
            matches = []
            for h_idx, h_data in enumerate(hands_data):
                best_dist = 400 
                best_tid = None
                for t_id in track_ids:
                    dist = np.linalg.norm(self.tracks[t_id] - h_data["center"])
                    if dist < best_dist:
                        best_dist = dist; best_tid = t_id
                if best_tid is not None:
                    matches.append((best_dist, h_idx, best_tid))
            
            matches.sort(key=lambda x: x[0])
            used_hands = set(); used_tracks = set()
            for dist, h_idx, t_id in matches:
                if h_idx not in used_hands and t_id not in used_tracks:
                    hands_data[h_idx]["id"] = t_id
                    used_hands.add(h_idx); used_tracks.add(t_id)
                    self.tracks[t_id] = hands_data[h_idx]["center"]

        track_keys = set(self.tracks.keys())
        for h_data in hands_data:
            if h_data["id"] is None:
                if 0 not in track_keys: new_id = 0
                elif 1 not in track_keys: new_id = 1
                else: new_id = self.next_id; self.next_id += 1
                h_data["id"] = new_id
                self.tracks[new_id] = h_data["center"]
                track_keys.add(new_id)

        active_ids = {h["id"] for h in hands_data}
        self.tracks = {k: v for k, v in self.tracks.items() if k in active_ids}
        
        output_hands = []
        for h_data in hands_data:
            t_id = h_data["id"]; key = f"ID_{t_id}"
            current_lms_px = h_data["landmarks"]
            
            if key not in self.prev_landmarks:
                self.prev_landmarks[key] = current_lms_px
            else:
                prev_lms = self.prev_landmarks[key]
                dist_moved = np.linalg.norm(np.array(current_lms_px[0][:2]) - np.array(prev_lms[0][:2]))
                alpha = get_adaptive_alpha(dist_moved)
                smoothed_lms = []
                for i in range(len(current_lms_px)):
                    curr = current_lms_px[i]; prev = prev_lms[i]
                    smoothed_lms.append([alpha*curr[0] + (1-alpha)*prev[0],
                                        alpha*curr[1] + (1-alpha)*prev[1],
                                        alpha*curr[2] + (1-alpha)*prev[2]])
                self.prev_landmarks[key] = smoothed_lms
                h_data["landmarks"] = smoothed_lms
            output_hands.append(h_data)

        output_hands.sort(key=lambda x: x["id"])
        return output_hands

def is_pinched(landmarks):
    # Perfected 3D Pinch Detection: Perspective invariant
    p0, p9 = np.array(landmarks[0]), np.array(landmarks[9])
    hand_size_3d = np.linalg.norm(p0 - p9)
    if hand_size_3d < 1: return False
    
    p4, p8 = np.array(landmarks[4]), np.array(landmarks[8])
    pinch_dist_3d = np.linalg.norm(p4 - p8)
    return (pinch_dist_3d / hand_size_3d) < 0.48 # Optimized for 3D

def draw_skeleton(img, lms, color):
    # Subtle ghost skeleton for user feedback
    connections = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
                   (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17)]
    for start, end in connections:
        p1 = (int(lms[start][0]), int(lms[start][1]))
        p2 = (int(lms[end][0]), int(lms[end][1]))
        cv2.line(img, p1, p2, color, 1)

class ParticleShape:
    def __init__(self, size=200):
        self.size = size
        self.cached_points = None
        self.cached_divs = -1

    def project_points(self, center_pos, scale, rotation_matrix, divisions=1, view_distance=1000):
        idx_div = int(round(divisions))
        if self.cached_points is None or self.cached_divs != idx_div:
            self.cached_points = self.generate_points(idx_div)
            self.cached_divs = idx_div

        points_3d = self.cached_points
        rotated = points_3d @ rotation_matrix.T
        rotated *= scale
        z_coords = rotated[:, 2]
        depth_denom = view_distance + z_coords
        depth_denom[depth_denom <= 0] = 0.0001
        factors = view_distance / depth_denom
        x_2d = rotated[:, 0] * factors + center_pos[0]
        y_2d = rotated[:, 1] * factors + center_pos[1]
        return np.column_stack((x_2d, y_2d, z_coords))

class Cube(ParticleShape):
    def generate_points(self, divisions):
        div = max(1, int(divisions))
        step = self.size / div
        offset = self.size / 2.0
        points = []
        line_range = np.linspace(-offset, offset, 20)
        for i in range(div + 1):
            for j in range(div + 1):
                f_y, f_z = -offset + i * step, -offset + j * step
                p = np.zeros((len(line_range), 3)); p[:, 0] = line_range; p[:, 1] = f_y; p[:, 2] = f_z; points.append(p)
                f_x, f_z = -offset + i * step, -offset + j * step
                p = np.zeros((len(line_range), 3)); p[:, 0] = f_x; p[:, 1] = line_range; p[:, 2] = f_z; points.append(p)
                f_x, f_y = -offset + i * step, -offset + j * step
                p = np.zeros((len(line_range), 3)); p[:, 0] = f_x; p[:, 1] = f_y; p[:, 2] = line_range; points.append(p)
        return np.vstack(points)

class Sphere(ParticleShape):
    def generate_points(self, divisions):
        div = max(12, int(divisions * 15))
        points = []
        radius = self.size / 2.0
        # Fibonacci Sphere for perfect point distribution
        phi = math.pi * (3. - math.sqrt(5.)) 
        num_points = div * div // 4
        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2
            r = math.sqrt(1 - y * y)
            theta = phi * i
            x = math.cos(theta) * r
            z = math.sin(theta) * r
            points.append([x * radius, y * radius, z * radius])
        return np.array(points)

class Pyramid(ParticleShape):
    def generate_points(self, divisions):
        div = max(1, int(divisions))
        offset = self.size / 2.0
        points = []
        steps = np.linspace(-offset, offset, 25)
        # Base edges 
        for s in steps:
            points.append([s, -offset, offset]); points.append([s, -offset, -offset])
            points.append([offset, -offset, s]); points.append([-offset, -offset, s])
        # Slant edges
        apex = np.array([0, offset, 0])
        base_corners = [[offset, -offset, offset], [offset, -offset, -offset],
                        [-offset, -offset, offset], [-offset, -offset, -offset]]
        for corner in base_corners:
            c = np.array(corner)
            for t in np.linspace(0, 1, 25): points.append(c * (1-t) + apex * t)
        # Horizontal slices
        for i in range(1, div + 1):
            t = i / (div + 1); loop_offset = offset * (1 - t); loop_y = -offset + (offset * 2 * t)
            l_steps = np.linspace(-loop_offset, loop_offset, 20)
            for s in l_steps:
                points.append([s, loop_y, loop_offset]); points.append([s, loop_y, -loop_offset])
                points.append([loop_offset, loop_y, s]); points.append([-loop_offset, loop_y, s])
        return np.array(points)

class BifurSystem(ParticleShape):
    def __init__(self, size=200):
        super().__init__(size)
        self.num_particles = 10000 # Optimized for 60FPS
        self.dt = 0.05
        self.b = 0.20818
        self.spread = 0.0
        self.particles = (np.random.rand(self.num_particles, 3) - 0.5) * 10.0
        
    def project_points(self, center_pos, scale, rotation_matrix, divisions=1, view_distance=1000):
        effective_b = self.b * (1.0 - self.spread * 0.5)
        pos = self.particles
        
        # RK2 Integration (2x faster than RK4, maintains chaotic stability)
        k1_x = np.sin(pos[:, 1]) - effective_b * pos[:, 0]
        k1_y = np.sin(pos[:, 2]) - effective_b * pos[:, 1]
        k1_z = np.sin(pos[:, 0]) - effective_b * pos[:, 2]
        k1 = np.column_stack((k1_x, k1_y, k1_z))
        
        pos_k2 = pos + k1 * self.dt
        k2_x = np.sin(pos_k2[:, 1]) - effective_b * pos_k2[:, 0]
        k2_y = np.sin(pos_k2[:, 2]) - effective_b * pos_k2[:, 1]
        k2_z = np.sin(pos_k2[:, 0]) - effective_b * pos_k2[:, 2]
        k2 = np.column_stack((k2_x, k2_y, k2_z))
        
        self.particles += (k1 + k2) * (self.dt * 0.5)
        
        dists = np.linalg.norm(self.particles, axis=1, keepdims=True)
        mask = (dists > 0.01).flatten()
        self.particles[mask] += (self.particles[mask] / dists[mask]) * self.spread * 0.05
        
        out_of_bounds = dists.flatten() > 12.0
        num_out = np.sum(out_of_bounds)
        if num_out > 0:
            self.particles[out_of_bounds] = (np.random.rand(num_out, 3) - 0.5) * 8.0

        visual_scale = scale * (self.size / 6.0) 
        rotated = self.particles @ rotation_matrix.T
        rotated *= visual_scale
        z_coords = rotated[:, 2]
        depth_denom = view_distance + z_coords
        depth_denom[depth_denom <= 0] = 0.0001
        factors = view_distance / depth_denom
        x_2d = rotated[:, 0] * factors + center_pos[0]
        y_2d = rotated[:, 1] * factors + center_pos[1]
        
        return np.column_stack((x_2d, y_2d, z_coords, self.particles[:, 0]))

def get_rotation_matrix(pitch, yaw, roll):
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cz, sz = np.cos(roll),  np.sin(roll)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

# --- Command Console (Tkinter) ---

class CommandConsole:
    def __init__(self, command_queue, log_queue):
        self.command_queue = command_queue
        self.log_queue = log_queue
        self.root = tk.Tk()
        self.root.title("gesp3D Command Line")
        self.root.geometry("450x400")
        self.root.configure(bg="#f2f0eb") 
        
        header = tk.Frame(self.root, bg="#1E3932", height=50); header.pack(fill=tk.X)
        tk.Label(header, text="gesp3D CONTROL CONSOLE", bg="#1E3932", fg="#ffffff", font=("Helvetica", 12, "bold")).pack(pady=10)
        
        self.output = tk.Text(self.root, bg="#ffffff", fg="#1E3932", font=("Consolas", 10), height=15, width=55, relief=tk.FLAT, padx=10, pady=10)
        self.output.pack(padx=20, pady=10); self.output.insert(tk.END, "System Ready.\nType '/options' for help.\n"); self.output.config(state=tk.DISABLED)
        
        entry_frame = tk.Frame(self.root, bg="#f2f0eb"); entry_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(entry_frame, text=">", bg="#f2f0eb", fg="#006241", font=("Consolas", 12, "bold")).pack(side=tk.LEFT)
        self.entry = tk.Entry(entry_frame, bg="#ffffff", fg="#006241", font=("Consolas", 11), relief=tk.FLAT, insertbackground="#006241")
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5); self.entry.bind("<Return>", self.handle_enter)
        
        self.status = tk.Label(self.root, text="Sandbox: Active", bg="#f2f0eb", fg="#00754A", font=("Helvetica", 9)); self.status.pack(side=tk.BOTTOM, anchor=tk.W, padx=20, pady=5)
        self.root.after(10, self.check_log_queue)

    def log(self, message):
        self.output.config(state=tk.NORMAL); self.output.insert(tk.END, f"{message}\n"); self.output.see(tk.END); self.output.config(state=tk.DISABLED)

    def handle_enter(self, event):
        cmd = self.entry.get().strip()
        if not cmd: return
        self.entry.delete(0, tk.END); self.log(f"[USER]: {cmd}"); self.command_queue.put(cmd)

    def check_log_queue(self):
        try:
            while not self.log_queue.empty(): self.log(self.log_queue.get_nowait())
        except queue.Empty: pass
        self.root.after(10, self.check_log_queue)

    def run(self):
        self.root.mainloop()

# --- Sandbox Logic ---

def run_sandbox(command_queue, log_queue, stop_event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 60) # Request 60 FPS from hardware
    cap.set(3, WIDTH); cap.set(4, HEIGHT)
    if not cap.isOpened(): log_queue.put("[ERR]: Webcam not found."); return
    try: detector = HandTracker()
    except Exception as e: log_queue.put(f"[ERR]: Detector failed: {e}"); return

    bifur_sys = BifurSystem(size=250)
    shapes = {"cube": Cube(size=250), "sphere": Sphere(size=250), "pyramid": Pyramid(size=250)}
    current_mode = "shapes"
    current_shape_name = "cube"; active_shape = shapes[current_shape_name]
    curr_cx, curr_cy = WIDTH / 2.0, HEIGHT / 2.0
    curr_scale, curr_roll, curr_pitch, curr_yaw, curr_divs = 1.0, 0.0, 0.0, 0.0, 1.0 
    holding = False; base_dist, base_scale = 1.0, 1.0; prev_time = 0
    target_offset_pitch, target_offset_yaw = 0.0, 0.0
    frame_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    while not stop_event.is_set():
        try:
            while not command_queue.empty():
                cmd = command_queue.get_nowait().lower()
                if cmd == "/options": log_queue.put("[SYS]: /mode, /cube, /sphere, /pyramid, /side <side>, /reset, /exit")
                elif cmd == "/mode": log_queue.put("[SYS]: Available modes: shapes, bifur. Type '/mode <name>'")
                elif cmd.startswith("/mode "):
                    mode = cmd.split(" ")[1] if len(cmd.split(" ")) > 1 else ""
                    if mode == "bifur":
                        current_mode = "bifur"; current_shape_name = "bifur"; active_shape = bifur_sys
                        log_queue.put("[SYS]: BIFUR Mode Active")
                    elif mode == "shapes":
                        current_mode = "shapes"; current_shape_name = "cube"; active_shape = shapes["cube"]
                        log_queue.put("[SYS]: SHAPES Mode Active")
                    else: log_queue.put("[SYS]: Unknown mode. Use shapes or bifur.")
                elif current_mode == "shapes" and cmd == "/cube": current_shape_name = "cube"; active_shape = shapes["cube"]; log_queue.put("[SYS]: Cube Active")
                elif current_mode == "shapes" and cmd == "/sphere": current_shape_name = "sphere"; active_shape = shapes["sphere"]; log_queue.put("[SYS]: Sphere Active")
                elif current_mode == "shapes" and cmd in ["/pyramid", "/triangle"]: current_shape_name = "pyramid"; active_shape = shapes["pyramid"]; log_queue.put("[SYS]: Pyramid Active")
                elif cmd.startswith("/side "):
                    side = cmd.split(" ")[1] if len(cmd.split(" ")) > 1 else ""
                    if side == "front": target_offset_pitch, target_offset_yaw = 0, 0
                    elif side == "back": target_offset_pitch, target_offset_yaw = 0, math.pi
                    elif side == "top": target_offset_pitch, target_offset_yaw = math.pi/2, 0
                    elif side == "bottom": target_offset_pitch, target_offset_yaw = -math.pi/2, 0
                    elif side == "left": target_offset_pitch, target_offset_yaw = 0, math.pi/2
                    elif side == "right": target_offset_pitch, target_offset_yaw = 0, -math.pi/2
                    log_queue.put(f"[SYS]: Snapping {side}")
                elif cmd == "/reset": curr_roll, curr_pitch, curr_yaw, target_offset_pitch, target_offset_yaw, curr_scale = 0,0,0,0,0,1.0; log_queue.put("[SYS]: Reset")
                elif cmd == "/exit": stop_event.set()
        except queue.Empty: pass

        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        hands = detector.find_hands(img)
        
        if current_mode == "bifur":
            if len(hands) > 0:
                frame_buffer = np.zeros_like(frame_buffer)
            else:
                frame_buffer = cv2.addWeighted(frame_buffer, 0.85, np.zeros_like(frame_buffer), 0.15, 0)
            black_bg = frame_buffer.copy()
        else:
            black_bg = np.zeros_like(img)
            
        t_pitch, t_yaw, t_roll = curr_pitch * 0.95, curr_yaw * 0.95, curr_roll 
        t_cx, t_cy, t_scale, t_divs = curr_cx, curr_cy, curr_scale, 1.0

        if len(hands) >= 2:
            hands.sort(key=lambda x: x["center"][0])
            h_left, h_right = hands[0], hands[1]
            lms_l, lms_r = h_left["landmarks"], h_right["landmarks"]
            
            # Use stable palm centers for position
            palm_l = np.mean([lms_l[i] for i in [0, 5, 9, 13, 17]], axis=0)
            palm_r = np.mean([lms_r[i] for i in [0, 5, 9, 13, 17]], axis=0)
            
            # Use index tips for rotation (more intuitive 'pointing')
            tip_l, tip_r = lms_l[8], lms_r[8]
            
            pinched1, pinched2 = is_pinched(h_left["raw_landmarks"]), is_pinched(h_right["raw_landmarks"])
            
            t_cx, t_cy = (palm_l[0] + palm_r[0]) / 2.0, (palm_l[1] + palm_r[1]) / 2.0
            dy, dx = tip_r[1] - tip_l[1], tip_r[0] - tip_l[0]; t_roll = math.atan2(dy, dx)
            dist = math.hypot(dx, dy)
            if pinched1 or pinched2:
                if not holding: holding = True; base_dist = max(dist, 10); base_scale = curr_scale
                t_scale = (dist / base_dist) * base_scale
            else: holding = False; t_scale = curr_scale
            
            t_yaw = (tip_l[2] - tip_r[2]) * 0.05 
            t_pitch = ((tip_l[1] + tip_r[1]) / 2.0 - HEIGHT/2.0) * 0.005
            
            if abs(t_roll) < 0.04: t_roll = 0.0
            if abs(t_yaw) < 0.04: t_yaw = 0.0
            if abs(t_pitch) < 0.04: t_pitch = 0.0 
            if pinched1 and pinched2: t_divs = min(6, max(1, (dist - 100) / 40.0))
            
            if current_mode == "bifur":
                if pinched1: bifur_sys.spread = min(1.0, bifur_sys.spread + 0.05)
                else: bifur_sys.spread = max(0.0, bifur_sys.spread - 0.05)
            
            # Feedback Drawing
            draw_skeleton(black_bg, lms_l, (20, 20, 20))
            draw_skeleton(black_bg, lms_r, (20, 20, 20))
            cv2.line(black_bg, (int(tip_l[0]), int(tip_l[1])), (int(tip_r[0]), int(tip_r[1])), (50, 50, 50), 1)
            cv2.circle(black_bg, (int(palm_l[0]), int(palm_l[1])), 6, (0, 255, 0) if pinched1 else (0, 0, 150), -1)
            cv2.circle(black_bg, (int(palm_r[0]), int(palm_r[1])), 6, (0, 255, 0) if pinched2 else (0, 0, 150), -1)

        alpha = get_adaptive_alpha(math.hypot(t_cx - curr_cx, t_cy - curr_cy))
        curr_cx, curr_cy = alpha * t_cx + (1 - alpha) * curr_cx, alpha * t_cy + (1 - alpha) * curr_cy
        curr_scale = alpha * t_scale + (1 - alpha) * curr_scale
        diff_roll = t_roll - curr_roll
        if diff_roll > math.pi: diff_roll -= 2*math.pi
        elif diff_roll < -math.pi: diff_roll += 2*math.pi
        curr_roll += ROTATION_ALPHA * diff_roll
        curr_yaw = ROTATION_ALPHA * (t_yaw + target_offset_yaw) + (1 - ROTATION_ALPHA) * curr_yaw
        curr_pitch = ROTATION_ALPHA * (t_pitch + target_offset_pitch) + (1 - ROTATION_ALPHA) * curr_pitch
        curr_divs = 0.1 * t_divs + 0.9 * curr_divs
        
        rot_matrix = get_rotation_matrix(curr_pitch, curr_yaw, curr_roll)
        projected = active_shape.project_points((curr_cx, curr_cy), curr_scale, rot_matrix, divisions=curr_divs)
        
        if len(projected) > 0:
            xs, ys, zs = projected[:, 0].astype(int), projected[:, 1].astype(int), projected[:, 2]
            mask = (xs >= 0) & (xs < WIDTH) & (ys >= 0) & (ys < HEIGHT)
            v_xs, v_ys, v_zs = xs[mask], ys[mask], zs[mask]
            if len(v_xs) > 0:
                if current_mode == "bifur":
                    orig_xs = projected[:, 3][mask]
                    mix_factors = np.clip((orig_xs + 10.0) / 20.0, 0, 1)
                    color_A = np.array([0, 136, 255]) # Orange BGR
                    color_B = np.array([255, 255, 0]) # Cyan BGR
                    colors = color_A * (1 - mix_factors[:, None]) + color_B * mix_factors[:, None]
                    frame_buffer[v_ys, v_xs] = colors.astype(np.uint8)
                    black_bg = frame_buffer.copy()
                else:
                    z_min, z_max = v_zs.min(), v_zs.max()
                    if z_max == z_min: z_max += 1
                    bins = np.linspace(z_max, z_min, 12)
                    for i in range(len(bins)-1):
                        layer = (v_zs <= bins[i]) & (v_zs > bins[i+1])
                        if not np.any(layer): continue
                        avg_z = (bins[i] + bins[i+1]) / 2; val = int(max(0, min(255, 128 + avg_z * 0.5)))
                        color = (val, 255, 255 - val//2) if holding else (255, val, 255 - val)
                        for px, py in zip(v_xs[layer], v_ys[layer]): cv2.circle(black_bg, (px, py), 1 if avg_z > 0 else 2, color, -1)

        cv2.putText(black_bg, f"SHAPE: {current_shape_name.upper()}", (15, HEIGHT-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        fps = int(1/(time.time()-prev_time)) if prev_time else 0; prev_time = time.time()
        cv2.putText(black_bg, f"FPS: {fps}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        cv2.imshow(WINDOW_NAME, black_bg)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows(); os._exit(0)

def main():
    from threading import Event
    command_queue, log_queue, stop_event = queue.Queue(), queue.Queue(), Event()
    console = CommandConsole(command_queue, log_queue)
    Thread(target=run_sandbox, args=(command_queue, log_queue, stop_event), daemon=True).start()
    console.run()

if __name__ == "__main__": main()
