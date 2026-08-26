import os
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ページ基本設定 & カスタムCSS（可読性向上・ダークモード対応）
st.set_page_config(
    page_title="PITCHING KINETIC ANALYSIS",
    page_icon="⚾",
    layout="wide"
)

st.markdown("""
<style>
    /* 全体背景と標準テキスト */
    .stApp {
        background-color: #0e1117;
    }
    
    /* 読みにくくなっていたテキストの色を明瞭な白色に補正 */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #f0f2f6 !important;
    }

    /* Metric（数値を表示するパーツ）の文字色調整 */
    [data-testid="stMetricLabel"] {
        color: #b0b8c4 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #00E5FF !important;
        font-size: 2.2rem !important;
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

st.markdown('<div class="main-title">PITCHING KINETIC ANALYSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI Motion Capture & Ground Reaction / Translational Speed Tracker</div>', unsafe_allow_html=True)

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

    pelvis_speeds = []
    thorax_speeds = []
    frame_numbers = []

    prev_pelvis = None
    prev_thorax = None

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

                left_hip = np.array([landmarks[23].x, landmarks[23].y])
                right_hip = np.array([landmarks[24].x, landmarks[24].y])
                pelvis_pos = (left_hip + right_hip) / 2.0

                left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
                right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
                thorax_pos = (left_shoulder + right_shoulder) / 2.0

                if prev_pelvis is not None:
                    p_speed = np.linalg.norm(pelvis_pos - prev_pelvis) * fps * 10
                    t_speed = np.linalg.norm(thorax_pos - prev_thorax) * fps * 10
                else:
                    p_speed = 0.0
                    t_speed = 0.0

                prev_pelvis = pelvis_pos
                prev_thorax = thorax_pos

                pelvis_speeds.append(p_speed)
                thorax_speeds.append(t_speed)
            else:
                pelvis_speeds.append(0.0)
                thorax_speeds.append(0.0)

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
        max_p_speed = max(pelvis_speeds) if pelvis_speeds else 0
        max_t_speed = max(thorax_speeds) if thorax_speeds else 0

        st.metric("Pelvis Peak Speed (骨盤最高速度)", f"{max_p_speed:.2f} a.u.")
        st.metric("Thorax Peak Speed (胸郭最高速度)", f"{max_t_speed:.2f} a.u.")
        
        st.info("💡 上段：実映像＋関節判定\n💡 下段：足型プレート・ArUco風付き棒人間")

    # グラフ表示
    st.markdown("---")
    st.subheader("📈 Translational Speed (地面反力・移動速度グラフ)")

    fig = go.Figure()

    window_size = 3
    p_smooth = np.convolve(pelvis_speeds, np.ones(window_size)/window_size, mode='same')
    t_smooth = np.convolve(thorax_speeds, np.ones(window_size)/window_size, mode='same')

    fig.add_trace(go.Scatter(
        x=frame_numbers, y=p_smooth, mode='lines', 
        name='1. Pelvis Trans. Speed (骨盤)', line=dict(color='#00AAFF', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=frame_numbers, y=t_smooth, mode='lines', 
        name='2. Thorax Trans. Speed (胸郭)', line=dict(color='#FF3333', width=3)
    ))

    fig.update_layout(
        title="Translational Speed (m/s - AR Calibrated)",
        xaxis_title="Video Frame",
        yaxis_title="Speed (m/s)",
        template="plotly_dark",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)')
    )

    st.plotly_chart(fig, use_container_width=True)
