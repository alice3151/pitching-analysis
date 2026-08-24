import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import tempfile
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ページ基本設定（スマホ表示向け）
st.set_page_config(page_title="投球動作解析アプリ", layout="centered")
st.title("⚾️ 投球動作並進速度解析")

# 1. パラメータ設定UI
st.sidebar.header("撮影設定")
shooting_fps = st.sidebar.selectbox("撮影FPS", [240.0, 120.0, 60.0, 30.0], index=0)
marker_body_m = st.sidebar.number_input("体幹ARマーカーサイズ (m)", value=0.05, step=0.01)

# 2. 動画アップロード
uploaded_file = st.file_uploader("スマホで撮影した動画を選択してください", type=["mov", "mp4"])

if uploaded_file is not None:
    # 一時ファイルとして保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mov')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.info("解析を実行中...（数十秒かかります）")
    
    # ─── ここから既存の解析処理 ───
    MARKER_BODY_M = marker_body_m
    MARKER_ARM_M  = 0.03

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)

   import os
import urllib.request

# モデルファイルが手元にない場合、公式サーバーから自動ダウンロード
model_path = 'pose_landmarker_heavy.task'
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    slow_factor = shooting_fps / fps
    dt = 1.0 / (fps * slow_factor)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarks_list, raw_frames = [], []
    scale_body_per_frame, scale_arm_per_frame = [], []

    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        raw_frames.append(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco_detector.detectMarkers(gray)
        body_m_per_px, arm_m_per_px = None, None
        
        if ids is not None and len(corners) > 0:
            for idx, marker_id in enumerate(ids.ravel()):
                c = corners[idx][0]
                side_px = np.linalg.norm(c[0] - c[1])
                if side_px > 0:
                    if marker_id < 4: body_m_per_px = MARKER_BODY_M / side_px
                    else: arm_m_per_px = MARKER_ARM_M / side_px
                        
        scale_body_per_frame.append(body_m_per_px)
        scale_arm_per_frame.append(arm_m_per_px)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_img)
        pts = {}
        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            for idx, pt in enumerate(res.pose_landmarks[0]):
                pts[idx] = np.array([pt.x * orig_w, pt.y * orig_h])
        landmarks_list.append(pts)

        frame_count += 1
        if frame_count % 10 == 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    num_frames = len(landmarks_list)

    # 補完処理
    def fill_scales(arr):
        last_val = 0.001
        for i in range(len(arr)):
            if arr[i] is not None: last_val = arr[i]
            else: arr[i] = last_val
        return arr

    scale_body_per_frame = fill_scales(scale_body_per_frame)
    
    # 全フレーム中央値で固定
    valid_scales = [s for s in scale_body_per_frame if s > 0]
    if valid_scales:
        scale_body_per_frame = [np.median(valid_scales)] * num_frames

    # 物理量計算
    raw_pelvis = np.zeros(num_frames)
    raw_thorax = np.zeros(num_frames)
    leg_height_px = np.zeros(num_frames)

    for f in range(1, num_frames):
        p_c, p_p = landmarks_list[f], landmarks_list[f-1]
        s_body = scale_body_per_frame[f]

        if 23 in p_c and 24 in p_c and 23 in p_p and 24 in p_p:
            pelvis_c = (p_c[23] + p_c[24]) / 2.0
            pelvis_p = (p_p[23] + p_p[24]) / 2.0
            raw_pelvis[f] = (np.linalg.norm(pelvis_c - pelvis_p) * s_body) / dt

        if 11 in p_c and 12 in p_c and 11 in p_p and 12 in p_p:
            thorax_c = (p_c[11] + p_c[12]) / 2.0
            thorax_p = (p_p[11] + p_p[12]) / 2.0
            raw_thorax[f] = (np.linalg.norm(thorax_c - thorax_p) * s_body) / dt

        if 25 in p_c:
            leg_height_px[f] = -p_c[25][1]

    def clean_signal(arr):
        smoothed = savgol_filter(arr, 15, 2)
        smoothed[smoothed < 0] = 0.0
        return smoothed

    v_pelvis = clean_signal(raw_pelvis)
    v_thorax = clean_signal(raw_thorax)
    v_pelvis[500:] = 0
    v_thorax[500:] = 0

    # ピーク抽出
    search_start = int(num_frames * 0.1)
    search_end   = int(num_frames * 0.65)
    target_heights = leg_height_px[search_start:search_end]
    peaks, _ = find_peaks(target_heights, distance=30, prominence=20)

    leg_up_frame = search_start + (peaks[-1] if len(peaks) > 0 else np.argmax(target_heights))
    pelvis_v_at_legup = v_pelvis[leg_up_frame]

    # 結果表示
    st.success("解析が完了しました！")
    col1, col2 = st.columns(2)
    col1.metric("レッグアップ位置", f"Frame {leg_up_frame}")
    col2.metric("レッグアップ時骨盤速度", f"{pelvis_v_at_legup:.2f} m/s")

    # グラフ表示
    fig, ax = plt.subplots(figsize=(8, 4))
    frames = np.arange(num_frames)
    ax.plot(frames, v_pelvis, label='Pelvis Speed (m/s)', color='#1f77b4')
    ax.plot(frames, v_thorax, label='Thorax Speed (m/s)', color='#d62728')
    ax.axvline(leg_up_frame, color='cyan', linestyle=':', label='Leg-Up Peak')
    ax.set_xlim(0, num_frames)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()
    st.pyplot(fig)
