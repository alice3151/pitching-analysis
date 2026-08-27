import os
import tempfile
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter
import mediapipe as mp

# MediaPipe のモジュール参照（エラー回避の正しい記述）
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ページ基本設定 & カスタムCSS
st.set_page_config(
    page_title="PITCHING KINETIC & ROTATIONAL ANALYSIS",
    page_icon="⚾",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown { color: #f0f2f6 !important; }
    [data-testid="stMetricLabel"] { color: #b0b8c4 !important; font-size: 0.9rem !important; }
    [data-testid="stMetricValue"] { color: #00E5FF !important; font-size: 2.0rem !important; font-weight: 700 !important; }
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-title { color: #a0aec0 !important; font-size: 0.95rem; margin-bottom: 25px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">PITCHING KINETIC & ROTATIONAL ANALYSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI Motion Capture - Multi-Foot GRF, Speed & Rotational Velocity Tracker</div>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Analysis Settings")
show_markers = st.sidebar.checkbox("ArUco風関節マーカー表示", value=True)
show_grf = st.sidebar.checkbox("地面反力(GRF)ベクトル表示", value=True)

uploaded_file = st.file_uploader("📁 解析する投球動画を選択してください (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

# GRF描画ロジック（410F未満：右足、410F〜490F：左足接地時）
def draw_advanced_skeleton(img, landmarks, frame_idx, show_grf=True):
    h, w, _ = img.shape
    def get_pt(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    try:
        r_shoulder, l_shoulder = get_pt(12), get_pt(11)
        r_elbow, l_elbow = get_pt(14), get_pt(13)
        r_wrist, l_wrist = get_pt(16), get_pt(15)
        r_hip, l_hip = get_pt(24), get_pt(23)
        r_knee, l_knee = get_pt(26), get_pt(25)
        r_ankle, l_ankle = get_pt(28), get_pt(27)
        r_heel, l_heel = get_pt(30), get_pt(29)
        r_foot, l_foot = get_pt(32), get_pt(31)
        nose = get_pt(0)
    except:
        return

    col_body = (0, 165, 255)
    col_right = (0, 0, 255)
    col_left = (255, 255, 0)
    col_joint = (0, 255, 255)
    col_head = (255, 255, 0)

    lines = [
        (r_shoulder, l_shoulder, col_body, 4), (r_shoulder, r_hip, col_body, 4),
        (l_shoulder, l_hip, col_body, 4), (r_hip, l_hip, col_body, 4),
        (r_shoulder, r_elbow, col_right, 4), (r_elbow, r_wrist, col_right, 4),
        (l_shoulder, l_elbow, col_left, 4), (l_elbow, l_wrist, col_left, 4),
        (r_hip, r_knee, col_right, 4), (r_knee, r_ankle, col_right, 4),
        (l_hip, l_knee, col_left, 4), (l_knee, l_ankle, col_left, 4),
        (r_ankle, r_heel, col_right, 3), (r_heel, r_foot, col_right, 3), (r_foot, r_ankle, col_right, 3),
        (l_ankle, l_heel, col_left, 3), (l_heel, l_foot, col_left, 3), (l_foot, l_ankle, col_left, 3),
    ]

    for p1, p2, col, thick in lines:
        cv2.line(img, p1, p2, col, thick, cv2.LINE_AA)

    # 地面反力(GRF)描画ロジック
    if show_grf:
        if frame_idx < 410:
            foot_pt = r_ankle
            arrow_end = (foot_pt[0] - 10, foot_pt[1] - 70)
            cv2.arrowedLine(img, foot_pt, arrow_end, (0, 0, 255), 4, tipLength=0.25)
            cv2.putText(img, "GRF", (arrow_end[0] - 15, arrow_end[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif 410 <= frame_idx <= 490:
            foot_pt = l_ankle
            if foot_pt[1] > h * 0.6:
                arrow_end = (foot_pt[0] - 30, foot_pt[1] - 90)
                cv2.arrowedLine(img, foot_pt, arrow_end, (0, 0, 255), 4, tipLength=0.25)
                cv2.putText(img, "GRF", (arrow_end[0] - 15, arrow_end[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    shoulder_center = ((r_shoulder[0] + l_shoulder[0]) // 2, (r_shoulder[1] + l_shoulder[1]) // 2)
    head_radius = max(int(np.linalg.norm(np.array(r_shoulder) - np.array(l_shoulder)) * 0.4), 12)
    head_center = (nose[0], nose[1] - int(head_radius * 0.2))
    
    cv2.circle(img, head_center, head_radius, col_head, 3, cv2.LINE_AA)
    cv2.line(img, head_center, shoulder_center, col_body, 3, cv2.LINE_AA)

    joints = [r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist,
              r_hip, l_hip, r_knee, l_knee, r_ankle, l_ankle, r_foot, l_foot]
    for j in joints:
        cv2.circle(img, j, 7, col_joint, -1, cv2.LINE_AA)
        if show_markers:
            cv2.circle(img, j, 11, (255, 255, 255), 2, cv2.LINE_AA)

def calculate_rotation_angle(p1, p2):
    dx = p2.x - p1.x
    dz = p2.z - p1.z
    return np.degrees(np.arctan2(dz, dx))

def clean_signal(arr):
    if len(arr) < 15:
        return np.array(arr)
    smoothed = savgol_filter(arr, 15, 2)
    smoothed[smoothed < 0] = 0.0
    return smoothed

if uploaded_file is not None:
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.read())
        input_path = tfile.name

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = 854
    target_h = int(height * (target_w / width))
    out_w, out_h = target_w, target_h * 2

    output_path = os.path.join(tempfile.gettempdir(), "analyzed_output.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("⚡ 骨格解析＆地面反力ベクトル描画中...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    pelvis_trans_raw, thorax_trans_raw = [], []
    pelvis_rot_raw, thorax_rot_raw = [], []
    frame_numbers = []

    prev_pelvis_pos, prev_thorax_pos = None, None
    prev_pelvis_angle, prev_thorax_angle = None, None

    style_left_node = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4)
    style_left_edge = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_resized = cv2.resize(frame, (target_w, target_h))
            black_bg = np.zeros_like(frame_resized)
            image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            results = pose.process(image_rgb)
            frame_drawn = frame_resized.copy()

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                p_pos = (np.array([landmarks[23].x, landmarks[23].y]) + np.array([landmarks[24].x, landmarks[24].y])) / 2.0
                t_pos = (np.array([landmarks[11].x, landmarks[11].y]) + np.array([landmarks[12].x, landmarks[12].y])) / 2.0

                p_trans_speed = np.linalg.norm(p_pos - prev_pelvis_pos) * fps * 10 if prev_pelvis_pos is not None else 0.0
                t_trans_speed = np.linalg.norm(t_pos - prev_thorax_pos) * fps * 10 if prev_thorax_pos is not None else 0.0
                prev_pelvis_pos, prev_thorax_pos = p_pos, t_pos

                p_angle = calculate_rotation_angle(landmarks[23], landmarks[24])
                t_angle = calculate_rotation_angle(landmarks[11], landmarks[12])

                if prev_pelvis_angle is not None:
                    d_p = np.abs(p_angle - prev_pelvis_angle)
                    d_t = np.abs(t_angle - prev_thorax_angle)
                    if d_p > 180: d_p = 360 - d_p
                    if d_t > 180: d_t = 360 - d_t
                    p_rot_speed = d_p * fps
                    t_rot_speed = d_t * fps
                else:
                    p_rot_speed, t_rot_speed = 0.0, 0.0

                prev_pelvis_angle, prev_thorax_angle = p_angle, t_angle

                mp_drawing.draw_landmarks(frame_drawn, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, style_left_node, style_left_edge)
                draw_advanced_skeleton(black_bg, landmarks, current_frame, show_grf=show_grf)

            else:
                p_trans_speed, t_trans_speed = 0.0, 0.0
                p_rot_speed, t_rot_speed = 0.0, 0.0

            pelvis_trans_raw.append(p_trans_speed)
            thorax_trans_raw.append(t_trans_speed)
            pelvis_rot_raw.append(p_rot_speed)
            thorax_rot_raw.append(t_rot_speed)
            frame_numbers.append(current_frame)

            combined_frame = np.vstack((frame_drawn, black_bg))
            out.write(combined_frame)
            current_frame += 1
            if total_frames > 0: progress_bar.progress(min(current_frame / total_frames, 1.0))

    cap.release()
    out.release()
    status_text.text("✅ 解析完了！")
    progress_bar.empty()

    p_trans_smooth = clean_signal(pelvis_trans_raw)
    t_trans_smooth = clean_signal(thorax_trans_raw)
    p_rot_smooth = clean_signal(pelvis_rot_raw)
    t_rot_smooth = clean_signal(thorax_rot_raw)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📹 2段縦並び 骨格＆地面反力モーション動画")
        st.video(output_path)
    with col2:
        st.subheader("📊 解析サマリー")
        st.metric("Pelvis Trans. Speed (骨盤最高移動速度)", f"{max(p_trans_smooth):.2f} a.u.")
        st.metric("Thorax Trans. Speed (胸郭最高移動速度)", f"{max(t_trans_smooth):.2f} a.u.")
        st.metric("Pelvis Rot. Velocity (骨盤最高回旋速度)", f"{max(p_rot_smooth):.1f} deg/s")
        st.metric("Thorax Rot. Velocity (胸郭最高回旋速度)", f"{max(t_rot_smooth):.1f} deg/s")
        st.info("💡 地面反力（GRF）ベクトルが足元に表示されます。")

    st.markdown("---")
    st.subheader("📈 Translational Speed (移動速度・地面反力指標)")
    fig_trans = go.Figure()
    fig_trans.add_trace(go.Scatter(x=frame_numbers, y=p_trans_smooth, mode='lines', name='1. Pelvis Trans. Speed', line=dict(color='#00AAFF', width=3)))
    fig_trans.add_trace(go.Scatter(x=frame_numbers, y=t_trans_smooth, mode='lines', name='2. Thorax Trans. Speed', line=dict(color='#FF3333', width=3)))
    fig_trans.update_layout(title="Translational Speed", template="plotly_dark", height=380)
    st.plotly_chart(fig_trans, use_container_width=True)

    st.subheader("🔄 Rotational Velocity (回旋角速度グラフ)")
    fig_rot = go.Figure()
    fig_rot.add_trace(go.Scatter(x=frame_numbers, y=p_rot_smooth, mode='lines', name='1. Pelvis Rot. Velocity', line=dict(color='#00E5FF', width=3)))
    fig_rot.add_trace(go.Scatter(x=frame_numbers, y=t_rot_smooth, mode='lines', name='2. Thorax Rot. Velocity', line=dict(color='#FF5252', width=3)))
    fig_rot.update_layout(title="Rotational Velocity", template="plotly_dark", height=380)
    st.plotly_chart(fig_rot, use_container_width=True)
