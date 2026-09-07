import os
import tempfile
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 骨格描画スタイル
LANDMARK_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=4)
CONNECTION_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=2)

st.set_page_config(
    page_title="PITCHING KINETIC & ROTATIONAL ANALYSIS",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ ピッチング動作・運動力学解析")

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

uploaded_file = st.file_uploader("動画ファイルをアップロードしてください (MP4 / MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out_overlay_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out_skeleton_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_overlay = cv2.VideoWriter(out_overlay_path, fourcc, orig_fps, (width, height))
    out_skeleton = cv2.VideoWriter(out_skeleton_path, fourcc, orig_fps, (width, height))

    st.info("動画を解析・生成中...")
    progress_bar = st.progress(0)

    wrist_history = []
    hip_velocities = []
    time_stamps = []
    prev_hip_x = None
    prev_wrist_pt = None
    
    # リリース判定フラグ
    tracking_active = True
    max_wrist_speed = 0.0
    speed_threshold = 20.0
    has_accelerated = False

    frame_count = 0

    is_right = (dominant_hand == "右投げ")
    wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST if is_right else mp_pose.PoseLandmark.LEFT_WRIST
    # 軸足（右投げ＝右足首、左投げ＝左足首）
    pivot_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE if is_right else mp_pose.PoseLandmark.LEFT_ANKLE

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 骨盤中心 (Mid-Hip)
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                hip_x = int(((l_hip.x + r_hip.x) / 2.0) * width)
                hip_y = int(((l_hip.y + r_hip.y) / 2.0) * height)

                # 並進速度計算
                if prev_hip_x is not None:
                    dx = abs(hip_x - prev_hip_x)
                    vel = dx * fps
                    hip_velocities.append(vel)
                    time_stamps.append(current_time)
                prev_hip_x = hip_x

                # 手首位置と移動速度（リリース検知用）
                wrist = landmarks[wrist_idx]
                wrist_pt = (int(wrist.x * width), int(wrist.y * height))

                if prev_wrist_pt is not None:
                    wrist_speed = np.sqrt((wrist_pt[0] - prev_wrist_pt[0])**2 + (wrist_pt[1] - prev_wrist_pt[1])**2)
                    if wrist_speed > speed_threshold:
                        has_accelerated = True
                        if wrist_speed > max_wrist_speed:
                            max_wrist_speed = wrist_speed

                    if has_accelerated and (wrist_speed < max_wrist_speed * 0.4):
                        tracking_active = False

                prev_wrist_pt = wrist_pt

                if tracking_active:
                    wrist_history.append(wrist_pt)

                # --- モード別描画 ---
                if analysis_mode == "テイクバック軌道追跡 (手首)":
                    for i in range(1, len(wrist_history)):
                        cv2.line(frame, wrist_history[i-1], wrist_history[i], (0, 0, 255), 4)
                        cv2.line(black_frame, wrist_history[i-1], wrist_history[i], (0, 0, 255), 4)
                    
                    if len(wrist_history) > 0:
                        cv2.circle(frame, wrist_history[-1], 6, (0, 0, 255), -1)
                        cv2.circle(black_frame, wrist_history[-1], 6, (0, 0, 255), -1)

                elif analysis_mode == "簡易地面反力 (GRF) 推定":
                    # 軸足の座標
                    ankle = landmarks[pivot_ankle_idx]
                    ankle_pt = (int(ankle.x * width), int(ankle.y * height))

                    # 軸足から骨盤（重心）へ向かうベクトル
                    vec_x = hip_x - ankle_pt[0]
                    vec_y = hip_y - ankle_pt[1]

                    # 地面反力の矢印終点（軸足から重心方向に伸びる）
                    # 視認しやすいようスケール倍率（1.2倍）をかけて足元から伸ばす
                    arrow_end = (int(ankle_pt[0] + vec_x * 1.2), int(ankle_pt[1] + vec_y * 1.2))

                    # 軸足から上・前方に向かう黄色い矢印を描画
                    cv2.arrowedLine(frame, ankle_pt, arrow_end, (0, 255, 255), 4, tipLength=0.25)
                    cv2.arrowedLine(black_frame, ankle_pt, arrow_end, (0, 255, 255), 4, tipLength=0.25)

                elif analysis_mode == "骨盤並進 (重心) 強調":
                    cv2.circle(frame, (hip_x, hip_y), 12, (255, 0, 0), -1)
                    cv2.circle(black_frame, (hip_x, hip_y), 12, (255, 0, 0), -1)

                # 骨格線の描画
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=LANDMARK_STYLE,
                    connection_drawing_spec=CONNECTION_STYLE
                )
                mp_drawing.draw_landmarks(
                    black_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=LANDMARK_STYLE,
                    connection_drawing_spec=CONNECTION_STYLE
                )

            out_overlay.write(frame)
            out_skeleton.write(black_frame)
            
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out_overlay.release()
    out_skeleton.release()

    st.success("解析処理が完了しました！")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 実動画 + 解析描画")
        st.video(out_overlay_path)
        with open(out_overlay_path, "rb") as f:
            st.download_button("📹 解析動画をダウンロード", f, file_name="analyzed_overlay.mp4", mime="video/mp4")

    with col2:
        st.subheader("🦴 骨格データ (ブラックスクリーン)")
        st.video(out_skeleton_path)
        with open(out_skeleton_path, "rb") as f:
            st.download_button("🦴 骨格動画をダウンロード", f, file_name="analyzed_skeleton.mp4", mime="video/mp4")

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
