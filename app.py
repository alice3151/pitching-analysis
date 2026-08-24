import os
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="投球フォーム解析", layout="wide")

st.title("投球フォーム 2画面骨格解析＆スピードグラフ")
st.write("動画をアップロードすると、骨格を描画した2画面動画と動作スピードのグラフを生成します。")

# サイドバー設定
bg_color_choice = st.sidebar.radio("右画面の背景色", ["黒 (Black)", "白 (White)"], index=0)
is_black_bg = bg_color_choice == "黒 (Black)"

# MediaPipe の初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

uploaded_file = st.file_uploader(
    "動画を選択してください (mp4, mov, avi)", type=["mp4", "mov", "avi"]
)

# カスタムスティック描画関数
def draw_custom_skeleton(img, landmarks, is_black=True):
    h, w, _ = img.shape
    
    # 座標変換関数
    def get_pt(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    # 主要関節座標
    try:
        r_shoulder, l_shoulder = get_pt(12), get_pt(11)
        r_elbow, l_elbow = get_pt(14), get_pt(13)
        r_wrist, l_wrist = get_pt(16), get_pt(15)
        r_hip, l_hip = get_pt(24), get_pt(23)
        r_knee, l_knee = get_pt(26), get_pt(25)
        r_ankle, l_ankle = get_pt(28), get_pt(27)
        nose = get_pt(0)
    except:
        return

    # 色の定義 (BGR)
    color_body = (0, 165, 255) if is_black else (255, 140, 0)   # 胴体（オレンジ）
    color_right = (0, 0, 255) if is_black else (200, 0, 0)      # 右側（赤）
    color_left = (255, 255, 0) if is_black else (0, 150, 255)   # 左側（シアン/青）
    color_joint = (0, 255, 255) if is_black else (0, 200, 200) # 関節（黄色）
    color_head = (255, 255, 0) if is_black else (100, 100, 255) # 頭部（薄青/黄色）

    # 1. 骨格線の描画 (Line)
    lines = [
        # 胴体
        (r_shoulder, l_shoulder, color_body, 4),
        (r_shoulder, r_hip, color_body, 4),
        (l_shoulder, l_hip, color_body, 4),
        (r_hip, l_hip, color_body, 4),
        # 右腕
        (r_shoulder, r_elbow, color_right, 4),
        (r_elbow, r_wrist, color_right, 4),
        # 左腕
        (l_shoulder, l_elbow, color_left, 4),
        (l_elbow, l_wrist, color_left, 4),
        # 右脚
        (r_hip, r_knee, color_right, 4),
        (r_knee, r_ankle, color_right, 4),
        # 左脚
        (l_hip, l_knee, color_left, 4),
        (l_knee, l_ankle, color_left, 4),
    ]

    for p1, p2, col, thick in lines:
        cv2.line(img, p1, p2, col, thick, cv2.LINE_AA)

    # 2. 頭部（大きめの円）の描画
    shoulder_center = ((r_shoulder[0] + l_shoulder[0]) // 2, (r_shoulder[1] + l_shoulder[1]) // 2)
    head_radius = int(np.linalg.norm(np.array(r_shoulder) - np.array(l_shoulder)) * 0.45)
    head_radius = max(head_radius, 12)
    head_center = (nose[0], nose[1] - int(head_radius * 0.3))
    
    cv2.circle(img, head_center, head_radius, color_head, 3, cv2.LINE_AA)
    # 首の接続
    cv2.line(img, head_center, shoulder_center, color_body, 3, cv2.LINE_AA)

    # 3. 関節（丸点）の描画
    joints = [r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist,
              r_hip, l_hip, r_knee, l_knee, r_ankle, l_ankle]
    for j in joints:
        cv2.circle(img, j, 6, color_joint, -1, cv2.LINE_AA)


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

    target_w = 640
    target_h = int(height * (target_w / width))
    out_w = target_w * 2
    out_h = target_h

    output_path = os.path.join(tempfile.gettempdir(), "analyzed_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("動画を解析・生成中...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    pelvis_speeds = []
    thorax_speeds = []
    frame_numbers = []

    prev_pelvis = None
    prev_thorax = None

    # 左画面用描画スタイル
    style_left_node = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=5)
    style_left_edge = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4)

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
            
            # 背景の作成（選択に応じて黒または白）
            if is_black_bg:
                bg_img = np.zeros_like(frame_resized)
            else:
                bg_img = np.full_like(frame_resized, 255)

            image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            results = pose.process(image_rgb)
            frame_drawn = frame_resized.copy()

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 左画面：標準骨格描画
                mp_drawing.draw_landmarks(
                    frame_drawn,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    style_left_node,
                    style_left_edge,
                )

                # 右画面：参考動画風カスタムスティックモデル
                draw_custom_skeleton(bg_img, landmarks, is_black=is_black_bg)

                # 速度算出
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

            combined_frame = np.hstack((frame_drawn, bg_img))
            out.write(combined_frame)

            current_frame += 1
            if total_frames > 0:
                progress_bar.progress(min(current_frame / total_frames, 1.0))

    cap.release()
    out.release()

    status_text.text("解析完了！")
    progress_bar.empty()

    # 1. 動画表示＆ダウンロード
    st.subheader("解析動画（2画面スティックモデル）")
    st.video(output_path)

    with open(output_path, "rb") as video_file:
        st.download_button(
            label="解析動画をダウンロード (MP4)",
            data=video_file,
            file_name="pitching_analysis.mp4",
            mime="video/mp4",
        )

    # 2. グラフ表示（ノイズを抑えた平滑化グラフ）
    st.markdown("---")
    st.subheader("Translational Speed (移動速度グラフ)")

    fig = go.Figure()

    # 移動平均フィルタでグラフのノイズ・ギザギザを軽減
    window_size = 5
    p_smooth = np.convolve(pelvis_speeds, np.ones(window_size)/window_size, mode='same')
    t_smooth = np.convolve(thorax_speeds, np.ones(window_size)/window_size, mode='same')

    fig.add_trace(go.Scatter(
        x=frame_numbers, y=p_smooth, mode='lines', 
        name='1. Pelvis Trans. Speed', line=dict(color='#00AAFF', width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=frame_numbers, y=t_smooth, mode='lines', 
        name='2. Thorax Trans. Speed', line=dict(color='#FF3333', width=2.5)
    ))

    fig.update_layout(
        title="体幹・腰の移動速度チャート（ノイズ除去済み）",
        xaxis_title="Video Frame",
        yaxis_title="Speed (a.u.)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(x=0.01, y=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)
