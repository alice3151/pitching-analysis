import os
import tempfile
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter

# MediaPipe のインポート
import mediapipe as mp
from mediapipe.solutions import pose as mp_pose
from mediapipe.solutions import drawing_utils as mp_drawing

# --------------------------------------------------
# ページ基本設定
# --------------------------------------------------
st.set_page_config(
    page_title="PITCHING KINETIC & ROTATIONAL ANALYSIS",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ ピッチング動作・運動力学解析アプリ")

# --------------------------------------------------
# サイドバー設定（モード & FPS設定）
# --------------------------------------------------
st.sidebar.header("⚙️ 解析設定")

# 1. モード選択
analysis_mode = st.sidebar.radio(
    "解析モード選択",
    ["テイクバック軌道追跡 (投げ手)", "簡易地面反力 (GRF) 推定", "骨盤並進 (重心) 速度解析"]
)

# 2. 利き腕選択
dominant_hand = st.sidebar.radio("投手タイプ", ["右投げ", "左投げ"])

# 3. 動画スピード設定（通常/スロー補正）
video_fps_mode = st.sidebar.selectbox(
    "撮影スピード設定",
    ["通常撮影 (30 fps - 試合等)", "スロー撮影 (60 fps)", "ハイスピード (120 fps)", "超スロー (240 fps)"]
)

fps_map = {
    "通常撮影 (30 fps - 試合等)": 30,
    "スロー撮影 (60 fps)": 60,
    "ハイスピード (120 fps)": 120,
    "超スロー (240 fps)": 240
}
fps = fps_map[video_fps_mode]

# --------------------------------------------------
# 3D 骨格描画関数
# --------------------------------------------------
def create_3d_skeleton(world_landmarks):
    if not world_landmarks:
        return None
    
    xs = [lm.x for lm in world_landmarks.landmark]
    ys = [-lm.z for lm in world_landmarks.landmark]
    zs = [-lm.y for lm in world_landmarks.landmark]

    fig = go.Figure()

    # 関節ノード
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=3, color='yellow')
    ))

    # 骨格エッジ
    for conn in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = conn
        fig.add_trace(go.Scatter3d(
            x=[xs[start_idx], xs[end_idx]],
            y=[ys[start_idx], ys[end_idx]],
            z=[zs[start_idx], zs[end_idx]],
            mode='lines',
            line=dict(color='cyan', width=4),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        paper_bgcolor="black",
        margin=dict(l=0, r=0, b=0, t=0),
        height=400
    )
    return fig

# --------------------------------------------------
# メイン処理：動画アップロードと解析
# --------------------------------------------------
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください (MP4 / MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    
    st_frame = st.empty()
    st_3d = st.empty()

    # 解析用変数の初期化
    wrist_history = []
    hip_velocities = []
    prev_hip_x = None

    # 関節インデックス設定
    is_right = (dominant_hand == "右投げ")
    wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST if is_right else mp_pose.PoseLandmark.LEFT_WRIST
    pivot_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE if is_right else mp_pose.PoseLandmark.LEFT_ANKLE

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # --- 骨盤中心 (Mid-Hip) 座標取得（マーカーレス並進計測） ---
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                hip_x = int(((l_hip.x + r_hip.x) / 2.0) * w)
                hip_y = int(((l_hip.y + r_hip.y) / 2.0) * h)

                # 並進速度計算 (ピクセル/秒)
                if prev_hip_x is not None:
                    dx = abs(hip_x - prev_hip_x)
                    velocity = dx * fps
                    hip_velocities.append(velocity)
                prev_hip_x = hip_x

                # --- モード 1: テイクバック手首軌道追跡 ---
                if analysis_mode == "テイクバック軌道追跡 (投げ手)":
                    wrist = landmarks[wrist_idx]
                    wrist_pt = (int(wrist.x * w), int(wrist.y * h))
                    wrist_history.append(wrist_pt)

                    # 軌跡描画 (赤ライン)
                    for i in range(1, len(wrist_history)):
                        cv2.line(frame, wrist_history[i-1], wrist_history[i], (0, 0, 255), 3)
                    cv2.circle(frame, wrist_pt, 6, (0, 255, 255), -1)

                # --- モード 2: 簡易地面反力 (GRF) 推定 ---
                elif analysis_mode == "簡易地面反力 (GRF) 推定":
                    ankle = landmarks[pivot_ankle_idx]
                    ankle_pt = (int(ankle.x * w), int(ankle.y * h))

                    # 軸足から骨盤（重心）へ向かうベクトル (GRF推定値)
                    grf_x = hip_x - ankle_pt[0]
                    grf_y = hip_y - ankle_pt[1]

                    # 軸足の位置から反力ベクトルを黄色の矢印で描画
                    arrow_end = (ankle_pt[0] + grf_x, ankle_pt[1] + grf_y)
                    cv2.arrowedLine(frame, ankle_pt, arrow_end, (0, 255, 255), 4, tipLength=0.2)
                    cv2.putText(frame, "Est. GRF Vector", (ankle_pt[0] - 20, ankle_pt[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # --- モード 3: 骨盤並進速度表示 ---
                elif analysis_mode == "骨盤並進 (重心) 速度解析":
                    cv2.circle(frame, (hip_x, hip_y), 8, (255, 0, 0), -1)
                    if len(hip_velocities) > 0:
                        cv2.putText(frame, f"Translation Speed: {int(hip_velocities[-1])} px/s", 
                                    (hip_x + 15, hip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # 通常の骨格ワイヤーフレームも描画
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 画面への描画出力
            st_frame.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    st.success("解析が完了しました！")
