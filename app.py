import os
import tempfile
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --------------------------------------------------
# ページ基本設定
# --------------------------------------------------
st.set_page_config(
    page_title="PITCHING KINETIC & ROTATIONAL ANALYSIS",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ ピッチング動作・運動力学解析")

# --------------------------------------------------
# サイドバー設定
# --------------------------------------------------
st.sidebar.header("⚙️ 解析・表示設定")

analysis_mode = st.sidebar.radio(
    "解析モード選択",
    ["標準 (骨格＆オーバーレイ)", "テイクバック軌道追跡 (手首)", "簡易地面反力 (GRF) 推定", "骨盤並進 (重心) 強調"]
)

dominant_hand = st.sidebar.radio("投手タイプ", ["右投げ", "左投げ"])

video_fps_mode = st.sidebar.selectbox(
    "撮影スピード設定",
    ["通常撮影 (30 fps)", "スロー撮影 (60 fps)", "ハイスピード (120 fps)", "超スロー (240 fps)"]
)
fps_map = {"通常撮影 (30 fps)": 30, "スロー撮影 (60 fps)": 60, "ハイスピード (120 fps)": 120, "超スロー (240 fps)": 240}
fps = fps_map[video_fps_mode]

# --------------------------------------------------
# 動画アップロード & 解析
# --------------------------------------------------
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください (MP4 / MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📹 実動画 + 解析描画")
        st_video = st.empty()
    with col2:
        st.subheader("🦴 骨格データ (ブラックスクリーン)")
        st_skeleton = st.empty()

    # データの蓄積用リスト
    wrist_history = []
    hip_velocities = []
    time_stamps = []
    prev_hip_x = None
    frame_count = 0

    is_right = (dominant_hand == "右投げ")
    wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST if is_right else mp_pose.PoseLandmark.LEFT_WRIST
    pivot_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE if is_right else mp_pose.PoseLandmark.LEFT_ANKLE

    progress_bar = st.progress(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps
            h, w, _ = frame.shape
            black_frame = np.zeros((h, w, 3), dtype=np.uint8)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 骨盤中心の計算
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                hip_x = int(((l_hip.x + r_hip.x) / 2.0) * w)
                hip_y = int(((l_hip.y + r_hip.y) / 2.0) * h)

                if prev_hip_x is not None:
                    dx = abs(hip_x - prev_hip_x)
                    vel = dx * fps
                    hip_velocities.append(vel)
                    time_stamps.append(current_time)
                prev_hip_x = hip_x

                # モード別描画
                if analysis_mode == "テイクバック軌道追跡 (手首)":
                    wrist = landmarks[wrist_idx]
                    wrist_pt = (int(wrist.x * w), int(wrist.y * h))
                    wrist_history.append(wrist_pt)
                    for i in range(1, len(wrist_history)):
                        cv2.line(frame, wrist_history[i-1], wrist_history[i], (0, 0, 255), 3)
                        cv2.line(black_frame, wrist_history[i-1], wrist_history[i], (0, 0, 255), 3)

                elif analysis_mode == "簡易地面反力 (GRF) 推定":
                    ankle = landmarks[pivot_ankle_idx]
                    ankle_pt = (int(ankle.x * w), int(ankle.y * h))
                    grf_x = hip_x - ankle_pt[0]
                    grf_y = hip_y - ankle_pt[1]
                    arrow_end = (ankle_pt[0] + grf_x, ankle_pt[1] + grf_y)
                    cv2.arrowedLine(frame, ankle_pt, arrow_end, (0, 255, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(black_frame, ankle_pt, arrow_end, (0, 255, 255), 4, tipLength=0.2)

                elif analysis_mode == "骨盤並進 (重心) 強調":
                    cv2.circle(frame, (hip_x, hip_y), 10, (255, 0, 0), -1)
                    cv2.circle(black_frame, (hip_x, hip_y), 10, (255, 0, 0), -1)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(black_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 映像のみリアルタイム更新 (Warning対策として use_container_width は使用)
            st_video.image(frame, channels="BGR", use_container_width=True)
            st_skeleton.image(black_frame, channels="BGR", use_container_width=True)
            
            # 進捗バー
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    st.success("解析処理が完了しました！")

    # --- 動画の全フレーム解析後にグラフを一括描画 (高速・非点滅) ---
    if len(hip_velocities) > 1:
        st.subheader("📈 骨盤並進 (重心) 速度グラフ")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_stamps, 
            y=hip_velocities, 
            mode='lines', 
            name='並進速度 (px/s)', 
            line=dict(color='cyan', width=2)
        ))
        fig.update_layout(
            xaxis_title="時間 (秒)",
            yaxis_title="速度 (px/s)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
