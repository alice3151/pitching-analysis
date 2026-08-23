import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

# ==========================================
# 1. ARマーカー & 基本設定
# ==========================================
MARKER_BODY_M = 0.05  # 腰・体幹用: 5cm
MARKER_ARM_M  = 0.03  # 手・腕用: 3cm

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)

model_path = 'pose_landmarker_heavy.task'
if not os.path.exists(model_path):
    url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'
    urllib.request.urlretrieve(url, model_path)

video_path = '/Users/arisukitamura/Desktop/動作解析/IMG_8828.mov'
output_video_path = 'kinetic_chain_aruco_complete.mp4'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

POSE_CONNECTIONS = [
    (0, 7), (0, 8), (0, 11), (0, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32)
]
KEYPOINTS_TO_DRAW = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# ==========================================
# 2. 動画の全フレーム解析（ArUco + Pose）
# ==========================================
cap = cv2.VideoCapture(video_path)

# ★【必須】VideoWriter用の fps 変数を定義（動画ファイルの標準再生速度を取得）
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  

# ★【速度修正】iPhoneの240fpsスロー撮影（動画ファイル自体の再生速度との倍率比）
RECORDING_FPS = 240.0                      # 実際の撮影時FPS
SLOW_FACTOR  = RECORDING_FPS / fps         # 倍率（例: 240 / 30 = 8倍スロー）
dt           = 1.0 / (fps * SLOW_FACTOR)   # 1フレームあたりの実際の物理時間（秒）

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

landmarks_list = []
raw_frames = []
scale_body_per_frame = []
scale_arm_per_frame  = []

while True:
    ret, frame = cap.read()
    if not ret: break
    raw_frames.append(frame)
    
    # --- ARマーカー検出 ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    
    body_m_per_px = None
    arm_m_per_px  = None
    
    if ids is not None and len(corners) > 0:
        for idx, marker_id in enumerate(ids.ravel()):
            c = corners[idx][0]
            side_px = np.linalg.norm(c[0] - c[1])
            if side_px > 0:
                if marker_id < 4:
                    body_m_per_px = MARKER_BODY_M / side_px
                else:
                    arm_m_per_px = MARKER_ARM_M / side_px
                    
    scale_body_per_frame.append(body_m_per_px)
    scale_arm_per_frame.append(arm_m_per_px)

    # --- 骨盤・関節検出 ---
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_img)
    pts = {}
    if res.pose_landmarks and len(res.pose_landmarks) > 0:
        for idx, pt in enumerate(res.pose_landmarks[0]):
            pts[idx] = np.array([pt.x * orig_w, pt.y * orig_h])
    landmarks_list.append(pts)

cap.release()
num_frames = len(landmarks_list)

# 検出漏れ補完
def fill_scales(arr, default_val=0.001):
    last_val = default_val
    for i in range(len(arr)):
        if arr[i] is not None: last_val = arr[i]
        else: arr[i] = last_val
    return arr

scale_body_per_frame = fill_scales(scale_body_per_frame)
scale_arm_per_frame  = fill_scales(scale_arm_per_frame)

# ★【追加】マーカーサイズの全フレーム中央値（メディアン）を計算して固定化
valid_body_scales = [s for s in scale_body_per_frame if s is not None and s > 0]
if valid_body_scales:
    median_body_scale = np.median(valid_body_scales)
    scale_body_per_frame = np.array([median_body_scale] * num_frames)

valid_arm_scales = [s for s in scale_arm_per_frame if s is not None and s > 0]
if valid_arm_scales:
    median_arm_scale = np.median(valid_arm_scales)
    scale_arm_per_frame = np.array([median_arm_scale] * num_frames)

# ★【追加1】ARマーカーサイズ計算の微振動を平滑化（速度跳ね上がり防止）
scale_body_per_frame = np.convolve(scale_body_per_frame, np.ones(5)/5, mode='same')
scale_arm_per_frame  = np.convolve(scale_arm_per_frame, np.ones(5)/5, mode='same')

def get_pt(pts, idx):
    return pts[idx] if idx in pts else None

# ==========================================
# 3. 物理量の計算 (m/s)
# ==========================================
raw_pelvis = np.zeros(num_frames)
raw_thorax = np.zeros(num_frames)
raw_arm    = np.zeros(num_frames)
raw_hand   = np.zeros(num_frames)
leg_height_px = np.zeros(num_frames)

for f in range(1, num_frames):
    p_c, p_p = landmarks_list[f], landmarks_list[f-1]
    s_body = scale_body_per_frame[f]
    s_arm  = scale_arm_per_frame[f]

    # 骨盤並進速度 (m/s)
    if 23 in p_c and 24 in p_c and 23 in p_p and 24 in p_p:
        pelvis_c = (p_c[23] + p_c[24]) / 2.0
        pelvis_p = (p_p[23] + p_p[24]) / 2.0
        raw_pelvis[f] = (np.linalg.norm(pelvis_c - pelvis_p) * s_body) / dt

    # 胸郭並進速度 (m/s)
    if 11 in p_c and 12 in p_c and 11 in p_p and 12 in p_p:
        thorax_c = (p_c[11] + p_c[12]) / 2.0
        thorax_p = (p_p[11] + p_p[12]) / 2.0
        raw_thorax[f] = (np.linalg.norm(thorax_c - thorax_p) * s_body) / dt

    # 肘・手速度 (m/s)
    e_c, e_p = get_pt(p_c, 14), get_pt(p_p, 14)
    if e_c is not None and e_p is not None:
        raw_arm[f] = (np.linalg.norm(e_c - e_p) * s_arm) / dt

    h_c, h_p = get_pt(p_c, 16), get_pt(p_p, 16)
    if h_c is not None and h_p is not None:
        raw_hand[f] = (np.linalg.norm(h_c - h_p) * s_arm) / dt

    if 25 in p_c:
        leg_height_px[f] = -p_c[25][1]

def clean_signal(arr):
    smoothed = savgol_filter(arr, 15, 2)
    smoothed[smoothed < 0] = 0.0
    return smoothed

v_pelvis = clean_signal(raw_pelvis)
v_thorax = clean_signal(raw_thorax)
v_arm    = clean_signal(raw_arm)
v_hand   = clean_signal(raw_hand)

# ★【追加2】投球終了後（500フレーム以降）の認識ブレ・異常スパイクをカット
v_pelvis[500:] = 0
v_thorax[500:] = 0
v_arm[500:]    = 0
v_hand[500:]   = 0

# ==========================================
# 4. レッグアップ最頂点の抽出（二段階モーション対応）
# ==========================================
search_start = int(num_frames * 0.1)
search_end   = int(num_frames * 0.65)
target_heights = leg_height_px[search_start:search_end]

peaks, _ = find_peaks(target_heights, distance=30, prominence=20)

if len(peaks) > 0:
    selected_peak_idx = peaks[-1]
    leg_up_frame = search_start + selected_peak_idx
else:
    leg_up_frame = search_start + np.argmax(target_heights)

pelvis_v_at_legup = v_pelvis[leg_up_frame]

print(f" 検出された足上げピーク数: {len(peaks)} 回")
print(f" ➔ 採用したレッグアップ最頂点（2回目）: Frame {leg_up_frame}")
print(f" ➔ その時の骨盤並進速度: {pelvis_v_at_legup:.2f} m/s")

# ==========================================
# 5. グラフ描画関数（骨盤・胸郭の並進速度のみ）
# ==========================================
def draw_report_graph(current_f):
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    frames = np.arange(num_frames)
    ax.plot(frames, v_pelvis, color='#1f77b4', label='1. Pelvis Trans. Speed (m/s)', linewidth=2.5)
    ax.plot(frames, v_thorax, color='#d62728', label='2. Thorax Trans. Speed (m/s)', linewidth=2.5)
    
    ax.axvline(current_f, color='yellow', linestyle='--', linewidth=2.0)
    ax.axvline(leg_up_frame, color='cyan', linestyle=':', linewidth=1.5, label='Leg-Up Peak')

    ax.set_xlim(0, num_frames)
    max_val = max(np.max(v_pelvis), np.max(v_thorax), 1.0)
    ax.set_ylim(0, max_val * 1.2)
    
    ax.set_title("Translational Speed (m/s - AR Calibrated)", color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel("Speed (m/s)", color='white', fontsize=9)
    ax.set_xlabel("Video Frame", color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(True, color='#333333')
    ax.legend(loc='upper left', fontsize=8, facecolor='#222222', edgecolor='none', labelcolor='white')
    
    fig.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

# ==========================================
# 6. 動画書き出し & オーバーレイ描画
# ==========================================
WIN_W, WIN_H = 1280, 720
TOP_H, BOT_H = 440, 280
PANEL_W = 640

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (WIN_W, WIN_H))

for f in range(num_frames):
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    raw_frame = cv2.resize(raw_frames[f], (PANEL_W, TOP_H))
    canvas[0:TOP_H, 0:PANEL_W] = raw_frame
    
    skel_canvas = np.zeros((TOP_H, PANEL_W, 3), dtype=np.uint8)
    pts = landmarks_list[f]

    if len(pts) > 0:
        scale_x = PANEL_W / orig_w
        scale_y = TOP_H / orig_h

        def transform_pt(pt_2d):
            return (int(pt_2d[0] * scale_x), int(pt_2d[1] * scale_y))

        for p1, p2 in POSE_CONNECTIONS:
            if p1 in pts and p2 in pts:
                cv2.line(skel_canvas, transform_pt(pts[p1]), transform_pt(pts[p2]), (0, 220, 255), 2)

        for idx, pt in pts.items():
            if idx in KEYPOINTS_TO_DRAW:
                radius = 6 if idx == 0 else 4
                color = (255, 200, 0) if idx in [0, 7, 8] else (0, 0, 255)
                cv2.circle(skel_canvas, transform_pt(pt), radius, color, -1)

        # 地面反力 (GRF) 描画
        r_foot, l_foot = get_pt(pts, 28), get_pt(pts, 27)
        if f < 410 and r_foot is not None:
            foot_pt = transform_pt(r_foot)
            arrow_end = (foot_pt[0] - 10, foot_pt[1] - 70)
            cv2.arrowedLine(skel_canvas, foot_pt, arrow_end, (0, 0, 255), 3, tipLength=0.25)
        elif 410 <= f <= 490 and l_foot is not None:
            foot_pt = transform_pt(l_foot)
            if foot_pt[1] > TOP_H * 0.7:
                arrow_end = (foot_pt[0] - 30, foot_pt[1] - 90)
                cv2.arrowedLine(skel_canvas, foot_pt, arrow_end, (0, 0, 255), 4, tipLength=0.25)

    cv2.putText(skel_canvas, f"Leg-Up Peak Frame: {leg_up_frame} F", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(skel_canvas, f"Pelvis Speed at Leg-Up: {pelvis_v_at_legup:.2f} m/s", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    canvas[0:TOP_H, PANEL_W:WIN_W] = skel_canvas
    
    graph_img = draw_report_graph(f)
    canvas[TOP_H:WIN_H, 0:WIN_W] = cv2.resize(graph_img, (WIN_W, BOT_H))
    
    cv2.rectangle(canvas, (0, 0), (WIN_W, WIN_H), (100, 100, 100), 2)
    cv2.line(canvas, (PANEL_W, 0), (PANEL_W, TOP_H), (100, 100, 100), 2)
    cv2.line(canvas, (0, TOP_H), (WIN_W, TOP_H), (100, 100, 100), 2)

    out.write(canvas)

out.release()

print("=" * 60)
print("【解析完了】")
print(f" レッグアップ最頂点: Frame {leg_up_frame}")
print(f" レッグアップ時の骨盤並進速度: {pelvis_v_at_legup:.2f} m/s")
print(f" 動画出力先: {output_video_path}")
print("=" * 60)