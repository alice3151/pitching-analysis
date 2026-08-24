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

# MediaPipe の初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

uploaded_file = st.file_uploader(
    "動画を選択してください (mp4, mov, avi)", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
    # 1. 一時入力ファイルの作成
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.read())
        input_path = tfile.name

    cap = cv2.VideoCapture(input_path)

    # 動画情報の取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 1画面あたり幅 640px に拡大して太く見やすく表示
    target_w = 640
    target_h = int(height * (target_w / width))
    out_w = target_w * 2
    out_h = target_h

    # 一時出力動画ファイルの設定
    output_path = os.path.join(tempfile.gettempdir(), "analyzed_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("動画を解析・生成中...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 速度データ計算用の保持リスト
    pelvis_speeds = []
    thorax_speeds = []
    frame_numbers = []

    prev_pelvis = None
    prev_thorax = None

    # 太い線と大きめの関節用スタイルの定義
    style_left_node = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=5)
    style_left_edge = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4)

    style_right_node = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=6)
    style_right_edge = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4)

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

            # リサイズ
            frame_resized = cv2.resize(frame, (target_w, target_h))
            black_bg = np.zeros_like(frame_resized)

            # BGR -> RGB
            image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            results = pose.process(image_rgb)
            frame_drawn = frame_resized.copy()

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # --- 骨格描画（太い線・大きな点） ---
                mp_drawing.draw_landmarks(
                    frame_drawn,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    style_left_node,
                    style_left_edge,
                )

                mp_drawing.draw_landmarks(
                    black_bg,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    style_right_node,
                    style_right_edge,
                )

                # --- スピード解析データ算出 (腰・胸の簡易速度計算) ---
                # 骨盤（両腰の重心）
                left_hip = np.array([landmarks[23].x, landmarks[23].y])
                right_hip = np.array([landmarks[24].x, landmarks[24].y])
                pelvis_pos = (left_hip + right_hip) / 2.0

                # 胸郭（両肩の重心）
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

            # 2画面結合
            combined_frame = np.hstack((frame_drawn, black_bg))

            # 動画への書き込み (グラフは入れない)
            out.write(combined_frame)

            current_frame += 1
            if total_frames > 0:
                progress_bar.progress(min(current_frame / total_frames, 1.0))

    cap.release()
    out.release()

    status_text.text("解析完了！")
    progress_bar.empty()

    # --- 1. 動画再生とダウンロードボタン ---
    st.subheader("解析動画（グラフなし）")
    st.video(output_path)

    with open(output_path, "rb") as video_file:
        st.download_button(
            label="解析動画をダウンロード (MP4)",
            data=video_file,
            file_name="pitching_analysis.mp4",
            mime="video/mp4",
        )

    # --- 2. 下部にグラフを表示 ---
    st.markdown("---")
    st.subheader("Translational Speed (移動速度グラフ)")

    fig = go.Figure()

    # 移動平均で滑らかにする
    window_size = 3
    p_smooth = np.convolve(pelvis_speeds, np.ones(window_size)/window_size, mode='same')
    t_smooth = np.convolve(thorax_speeds, np.ones(window_size)/window_size, mode='same')

    fig.add_trace(go.Scatter(x=frame_numbers, y=p_smooth, mode='lines', name='1. Pelvis Trans. Speed', line=dict(color='#00AAFF', width=2)))
    fig.add_trace(go.Scatter(x=frame_numbers, y=t_smooth, mode='lines', name='2. Thorax Trans. Speed', line=dict(color='#FF3333', width=2)))

    fig.update_layout(
        title="体幹・腰の移動速度チャート",
        xaxis_title="Video Frame",
        yaxis_title="Speed (a.u.)",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(x=0.01, y=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)
