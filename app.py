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
    ["標準 (骨格＆オーバーレイ)", "テイクバック軌道追跡 (手首)", "物理ベース地面反力 (GRF) 推定", "骨盤並進 (重心) 強調"]
)

dominant_hand = st.sidebar.radio("投手タイプ", ["右投げ", "左投げ"])

video_fps_mode = st.sidebar.selectbox(
    "撮影スピード設定",
    ["通常撮影 (30 fps)", "スロー撮影 (60 fps)", "ハイスピード (120 fps)", "超スロー (240 fps)"]
)
fps_map = {"通常撮影 (30 fps)": 30, "スロー撮影 (60 fps)": 60, "ハイスピード (120 fps)": 120, "超スロー (240 fps)": 240}
fps = fps_map[video_fps_mode]

# 物理パラメータ設定
user_weight = st.sidebar.number_input("体重 (kg)", min_value=30.0, max_value=120.0, value=65.0, step=1.0)
GRAVITY = 9.81  # m/s^2

uploaded_file = st.file_uploader("動画ファイルをアップロードしてください (MP4 / MOV)", type=["mp4", "mov", "avi"])

# 3〜5フレームの移動平均フィルタ関数
def smooth_landmarks_history(history, window_size=5):
    if len(history) < 2:
        return history[-1]
    curr_window = history[-window_size:]
    avg_x = np.mean([pt[0] for pt in curr_window])
    avg_y = np.mean([pt[1] for pt in curr_window])
    return (avg_x, avg_y)

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

    # 履歴バッファ
    hip_raw_history = []
    wrist_raw_history = []
    pivot_ankle_raw_history = []
    lead_ankle_raw_history = []
    
    # 物理量計算用履歴
    com_y_m_history = []     # 重心Y座標 (m)
    com_vy_m_history = []    # 重心Y速度 (m/s)
    com_vx_m_history = []    # 重心X速度 (m/s)
    time_stamps = []
    
    wrist_display_history = []
    
    # リリース判定
    tracking_active = True
    max_wrist_speed = 0.0
    has_accelerated = False

    frame_count = 0
    dt = 1.0 / fps  # 1フレームあたりの秒数

    is_right = (dominant_hand == "右投げ")
    wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST if is_right else mp_pose.PoseLandmark.LEFT_WRIST
    pivot_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE if is_right else mp_pose.PoseLandmark.LEFT_ANKLE
    lead_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE if is_right else mp_pose.PoseLandmark.RIGHT_ANKLE

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
            current_time = frame_count * dt
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # --- 1. ピクセル座標の取得 ---
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                
                # 生座標 (px)
                raw_hip = (((l_hip.x + r_hip.x) / 2.0) * width, ((l_hip.y + r_hip.y) / 2.0) * height)
                raw_wrist = (landmarks[wrist_idx].x * width, landmarks[wrist_idx].y * height)
                raw_pivot = (landmarks[pivot_ankle_idx].x * width, landmarks[pivot_ankle_idx].y * height)
                raw_lead = (landmarks[lead_ankle_idx].x * width, landmarks[lead_ankle_idx].y * height)

                hip_raw_history.append(raw_hip)
                wrist_raw_history.append(raw_wrist)
                pivot_ankle_raw_history.append(raw_pivot)
                lead_ankle_raw_history.append(raw_lead)

                # --- 2. 移動平均フィルタによる平滑化 (過去5フレーム) ---
                hip_pt = smooth_landmarks_history(hip_raw_history, window_size=5)
                wrist_pt = smooth_landmarks_history(wrist_raw_history, window_size=5)
                pivot_pt = smooth_landmarks_history(pivot_ankle_raw_history, window_size=5)
                lead_pt = smooth_landmarks_history(lead_ankle_raw_history, window_size=5)

                # --- 3. px → m 変換スケールの計算 ---
                # 左右の股関節距離(px)を 成人平均 約0.18m とする
                hip_dx_px = (r_hip.x - l_hip.x) * width
                hip_dy_px = (r_hip.y - l_hip.y) * height
                hip_dist_px = np.sqrt(hip_dx_px**2 + hip_dy_px**2)
                
                # 検出エラー防止（最小px保護）
                scale_px_to_m = 0.18 / max(hip_dist_px, 15.0)

                # --- 4. 速度 (m/s) & 加速度 (m/s^2) & GRF計算 ---
                hip_x_m = hip_pt[0] * scale_px_to_m
                hip_y_m = hip_pt[1] * scale_px_to_m  # 画像座標系: 下方向がプラス

                grf_y_N = 0.0
                vx_m = 0.0

                if len(com_y_m_history) >= 1:
                    # 1次微分: 速度 (m/s)
                    vx_m = abs(hip_x_m - (com_vx_m_history[-1] if com_vx_m_history else hip_x_m)) / dt  # 簡易X速度
                    vy_m = (hip_y_m - com_y_m_history[-1]) / dt  # Y速度（下方向プラス）
                    
                    com_vx_m_history.append(vx_m)
                    
                    if len(com_vy_m_history) >= 1:
                        # 2次微分: 上下加速度 a_y (m/s^2)
                        # 画像座標系で下が正のため、鉛直上向き加速度は -d(vy)/dt
                        a_y = - (vy_m - com_vy_m_history[-1]) / dt

                        # 地面反力 F = m * (a_y + g)
                        # 重力加速度 9.81 を加算
                        grf_y_N = user_weight * max(0.0, (a_y + GRAVITY))
                        
                    com_vy_m_history.append(vy_m)
                else:
                    com_vx_m_history.append(0.0)
                    com_vy_m_history.append(0.0)

                com_y_m_history.append(hip_y_m)
                time_stamps.append(current_time)

                # リリリース判定 (手首速度に基づく)
                if len(wrist_raw_history) >= 2:
                    w_speed = np.sqrt((wrist_pt[0] - wrist_raw_history[-2][0])**2 + (wrist_pt[1] - wrist_raw_history[-2][1])**2) * scale_px_to_m / dt
                    if w_speed > 2.0:  # 2 m/s 以上で振りかぶり判定
                        has_accelerated = True
                        max_wrist_speed = max(max_wrist_speed, w_speed)
                    if has_accelerated and (w_speed < max_wrist_speed * 0.35):
                        tracking_active = False

                if tracking_active:
                    wrist_display_history.append((int(wrist_pt[0]), int(wrist_pt[1])))

                # --- 描画処理 ---
                int_hip = (int(hip_pt[0]), int(hip_pt[1]))
                int_pivot = (int(pivot_pt[0]), int(pivot_pt[1]))
                int_lead = (int(lead_pt[0]), int(lead_pt[1]))

                if analysis_mode == "テイクバック軌道追跡 (手首)":
                    for i in range(1, len(wrist_display_history)):
                        cv2.line(frame, wrist_display_history[i-1], wrist_display_history[i], (0, 0, 255), 4)
                        cv2.line(black_frame, wrist_display_history[i-1], wrist_display_history[i], (0, 0, 255), 4)

                elif analysis_mode == "物理ベース地面反力 (GRF) 推定":
                    if tracking_active:
                        # 着地足の判定（高さ比較）
                        active_foot = int_lead if lead_pt[1] >= (pivot_pt[1] - 10) else int_pivot

                        # GRFの大きさに応じた矢印長さを算出 (例: 1000N ≒ 100px)
                        arrow_len_px = int((grf_y_N / (user_weight * GRAVITY)) * 60)
                        arrow_len_px = min(max(arrow_len_px, 10), 180)  # 描画サイズ制限

                        # 接地足から上（鉛直上向き）へ向かう反力矢印
                        arrow_end = (active_foot[0], active_foot[1] - arrow_len_px)

                        cv2.arrowedLine(frame, active_foot, arrow_end, (0, 255, 255), 4, tipLength=0.3)
                        cv2.arrowedLine(black_frame, active_foot, arrow_end, (0, 255, 255), 4, tipLength=0.3)
                        
                        # 画面上にリアルタイムの力 (N) を表示
                        cv2.putText(frame, f"GRF: {int(grf_y_N)} N", (active_foot[0] + 15, active_foot[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                elif analysis_mode == "骨盤並進 (重心) 強調":
                    cv2.circle(frame, int_hip, 12, (255, 0, 0), -1)
                    cv2.circle(black_frame, int_hip, 12, (255, 0, 0), -1)

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

    with col2:
        st.subheader("🦴 骨格データ (ブラックスクリーン)")
        st.video(out_skeleton_path)

    # 単位を m/s にした正確な並進速度グラフ
    if len(com_vx_m_history) > 1:
        st.subheader("📈 骨盤並進 (重心) 速度グラフ")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_stamps, 
            y=com_vx_m_history, 
            mode='lines', 
            name='並進速度 (m/s)', 
            line=dict(color='cyan', width=2)
        ))
        fig.update_layout(
            xaxis_title="時間 (秒)",
            yaxis_title="速度 (m/s)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
