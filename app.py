import os
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ページ基本設定 & カスタムCSS（高視認性ダークモード）
st.set_page_config(
    page_title="PITCHING KINETIC & ROTATIONAL ANALYSIS",
    page_icon="⚾",
    layout="wide"
)

st.markdown("""
<style>
    /* 全体背景と標準テキスト */
    .stApp {
        background-color: #0e1117;
    }
    
    /* テキストの色を明瞭な白色に補正 */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #f0f2f6 !important;
    }

    /* Metric（数値パーツ）の文字色調整 */
    [data-testid="stMetricLabel"] {
        color: #b0b8c4 !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #00E5FF !important;
        font-size: 2.0rem !important;
        font-weight: 700 !important;
    }

    /* タイトルのグラデーション */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #a0aec0 !important;
        font-size: 0.95rem;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">PITCHING KINETIC & ROTATIONAL ANALYSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI Motion Capture - Translational Speed & Rotational Velocity Tracker</div>', unsafe_allow_html=True)

# MediaPipe Pose の初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# サイドバー設定
st.sidebar.header("⚙️ Analysis Settings")
show_markers = st.sidebar.checkbox("ArUco風関節マーカー表示", value=True)

uploaded_file = st.file_uploader(
    "📁 解析する投球動画を選択してください (MP4, MOV, AVI)", type=["mp4", "mov", "avi"]
)

# 骨格描画関数
def draw_advanced_skeleton(img, landmarks):
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
        (r_shoulder, l_shoulder, col_body, 4),
        (r_shoulder, r_hip, col_body, 4),
        (l_shoulder, l_hip, col_body, 4),
        (r_hip, l_hip, col_body, 4),
        (r_shoulder, r_elbow, col_right, 4),
        (r_elbow, r_wrist, col_right, 4),
        (l_shoulder, l_elbow, col_left, 4),
        (l_elbow, l_wrist, col_left, 4),
        (r_hip, r_knee, col_right, 4),
        (r_knee, r_ankle, col_right, 4),
        (l_hip, l_knee, col_left, 4),
        (l_knee, l_ankle, col_left, 4),
        (r_ankle, r_heel, col_right, 3),
        (r_heel, r_foot, col_right, 3),
        (r_foot, r_ankle, col_right, 3),
        (l_ankle, l_heel, col_left, 3),
        (l_heel, l_foot, col_left, 3),
        (l_foot, l_ankle, col_left, 3),
    ]

    for p1, p2, col, thick in lines:
        cv2.line(img, p1, p2, col, thick, cv2.LINE_AA)

    r_foot_poly = np.array([r_ankle, r_heel, r_foot], np.int32)
    l_foot_poly = np.array([l_ankle, l_heel, l_foot], np.int32)
    cv2.fillPoly(img, [r_foot_poly], (0, 0, 180))
    cv2.fillPoly(img, [l_foot_poly], (180, 180, 0))

    shoulder_center = ((r_shoulder[0] + l_shoulder[0]) // 2, (r_shoulder[1] + l_shoulder[1]) // 2)
    head_radius = int(np.linalg.norm(np.array(r_shoulder) - np.array(l_shoulder)) * 0.4)
    head_radius = max(head_radius, 12)
    head_center = (nose[0], nose[1] - int(head_radius * 0.2))
    
    cv2.circle(img, head_center, head_radius, col_head, 3, cv2.LINE_AA)
    cv2.line(img, head_center, shoulder_center, col_body, 3, cv2.LINE_AA)

    joints = [r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist,
              r_hip, l_hip, r_knee, l_knee, r_ankle, l_ankle, r_foot, l_foot]
    for j in joints:
        cv2.circle(img, j, 7, col_joint, -1, cv2.LINE_AA)
        if show_markers:
            cv2.circle(img, j, 11, (255, 255, 255), 2, cv2.LINE_AA)

# 回旋角度計算関数
def calculate_rotation_angle(p1, p2):
    dx = p2.x - p1.x
    dz = p2.z - p1.z
    angle_rad = np.arctan2(dz, dx)
    return np.degrees(angle_rad)


if uploaded_file is not None:
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.read())
        input_path = tfile.name

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = 854
    target_h = int(height * (target_w / width))
    out_w = target_w
    out_h = target_h * 2

    output_path = os.path.join(tempfile.gettempdir(), "analyzed_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("⚡ 骨格フレーム解析および動画生成中...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 移動速度用配列
    pelvis_trans_speeds = []
    thorax_trans_speeds = []

    # 回旋角速度用配列
    pelvis_rot_speeds = []
    thorax_rot_speeds = []

    frame_numbers = []

    prev_pelvis_pos = None
    prev_thorax_pos = None

    prev_pelvis_angle = None
    prev_thorax_angle = None

    style_left_node = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4)
    style_left_edge = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (target_w, target_h))
            black_bg = np.zeros_like(frame_resized)

            image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            results = pose.process(image_rgb)
            frame_drawn = frame_resized.copy()

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                mp_drawing.draw_landmarks(frame_drawn, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, style_left_node, style_left_edge)
                draw_advanced_skeleton(black_bg, landmarks)

                # --- 1. 移動速度（Translational Speed）算出 ---
                left_hip = np.array([landmarks[23].x, landmarks[23].y])
                right_hip = np.array([landmarks[24].x, landmarks[24].y])
                pelvis_pos = (left_hip + right_hip) / 2.0

                left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
                right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
                thorax_pos = (left_shoulder + right_shoulder) / 2.0

                if prev_pelvis_pos is not None:
                    p_trans_speed = np.linalg.norm(pelvis_pos - prev_pelvis_pos) * fps * 10
                    t_trans_speed = np.linalg.norm(thorax_pos - prev_thorax_pos) * fps * 10
                else:
                    p_trans_speed = 0.0
                    t_trans_speed = 0.0

                prev_pelvis_pos = pelvis_pos
                prev_thorax_pos = thorax_pos

                pelvis_trans_speeds.append(p_trans_speed)
                thorax_trans_speeds.append(t_trans_speed)

                # --- 2. 回旋角速度（Rotational Velocity）算出 ---
                pelvis_angle = calculate_rotation_angle(landmarks[23], landmarks[24])
                thorax_angle = calculate_rotation_angle(landmarks[11], landmarks[12])

                if prev_pelvis_angle is not None:
                    d_pelvis = np.abs(pelvis_angle - prev_pelvis_angle)
                    d_thorax = np.abs(thorax_angle - prev_thorax_angle)
                    
                    if d_pelvis > 180: d_pelvis = 360 - d_pelvis
                    if d_thorax > 180: d_thorax = 360 - d_thorax

                    p_rot_speed = d_pelvis * fps
                    t_rot_speed = d_thorax * fps
                else:
                    p_rot_speed = 0.0
                    t_rot_speed = 0.0

                prev_pelvis_angle = pelvis_angle
                prev_thorax_angle = thorax_angle

                pelvis_rot_speeds.append(p_rot_speed)
                thorax_rot_speeds.append(t_rot_speed)

            else:
                pelvis_trans_speeds.append(0.0)
                thorax_trans_speeds.append(0.0)
                pelvis_rot_speeds.append(0.0)
                thorax_rot_speeds.append(0.0)

            frame_numbers.append(current_frame)

            combined_frame = np.vstack((frame_drawn, black_bg))
            out.write(combined_frame)

            current_frame += 1
            if total_frames > 0:
                progress_bar.progress(min(current_frame / total_frames, 1.0))

    cap.release()
    out.release()

    status_text.text("✅ 解析完了！")
    progress_bar.empty()

    # レンダリングレイアウト
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 2段縦並び 骨格比較モーション動画")
        st.video(output_path)

        with open(output_path, "rb") as video_file:
            st.download_button(
                label="📥 解析動画をダウンロード (MP4)",
                data=video_file,
                file_name="pitching_analysis_vertical.mp4",
                mime="video/mp4",
            )

    with col2:
        st.subheader("📊 解析サマリー")
        max_p_trans = max(pelvis_trans_speeds) if pelvis_trans_speeds else 0
        max_t_trans = max(thorax_trans_speeds) if thorax_trans_speeds else 0
        max_p_rot = max(pelvis_rot_speeds) if pelvis_rot_speeds else 0
        max_t_rot = max(thorax_rot_speeds) if thorax_rot_speeds else 0

        st.metric("Pelvis Trans. Speed (骨盤最高移動速度)", f"{max_p_trans:.2f} a.u.")
        st.metric("Thorax Trans. Speed (胸郭最高移動速度)", f"{max_t_trans:.2f} a.u.")
        st.metric("Pelvis Rot. Velocity (骨盤最高回旋速度)", f"{max_p_rot:.1f} deg/s")
        st.metric("Thorax Rot. Velocity (胸郭最高回旋速度)", f"{max_t_rot:.1f} deg/s")
        
        st.info("💡 上段：実映像＋関節判定\n💡 下段：足型プレート・ArUco風付き棒人間")

    # --- グラフ描画（移動速度 & 回旋速度） ---
    st.markdown("---")
    
    # 1. 移動速度グラフ
    st.subheader("📈 Translational Speed (移動速度・地面反力指標)")
    fig_trans = go.Figure()

    window_size = 3
    p_trans_smooth = np.convolve(pelvis_trans_speeds, np.ones(window_size)/window_size, mode='same')
    t_trans_smooth = np.convolve(thorax_trans_speeds, np.ones(window_size)/window_size, mode='same')

    fig_trans.add_trace(go.Scatter(
        x=frame_numbers, y=p_trans_smooth, mode='lines', 
        name='1. Pelvis Trans. Speed (骨盤移動速度)', line=dict(color='#00AAFF', width=3)
    ))
    fig_trans.add_trace(go.Scatter(
        x=frame_numbers, y=t_trans_smooth, mode='lines', 
        name='2. Thorax Trans. Speed (胸郭移動速度)', line=dict(color='#FF3333', width=3)
    ))

    fig_trans.update_layout(
        title="Translational Speed (m/s - AR Calibrated)",
        xaxis_title="Video Frame",
        yaxis_title="Speed (a.u.)",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)')
    )
    st.plotly_chart(fig_trans, use_container_width=True)

    # 2. 回旋角速度グラフ
    st.subheader("🔄 Rotational Velocity (回旋角速度グラフ)")
    fig_rot = go.Figure()

    window_rot = 5
    p_rot_smooth = np.convolve(pelvis_rot_speeds, np.ones(window_rot)/window_rot, mode='same')
    t_rot_smooth = np.convolve(thorax_rot_speeds, np.ones(window_rot)/window_rot, mode='same')

    fig_rot.add_trace(go.Scatter(
        x=frame_numbers, y=p_rot_smooth, mode='lines', 
        name='1. Pelvis Rot. Velocity (骨盤回旋速度)', line=dict(color='#00E5FF', width=3)
    ))
    fig_rot.add_trace(go.Scatter(
        x=frame_numbers, y=t_rot_smooth, mode='lines', 
        name='2. Thorax Rot. Velocity (胸郭回旋速度)', line=dict(color='#FF5252', width=3)
    ))

    fig_rot.update_layout(
        title="Rotational Velocity (deg/s)",
        xaxis_title="Video Frame",
        yaxis_title="Angular Velocity (deg/s)",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)')
    )
    st.plotly_chart(fig_rot, use_container_width=True)
