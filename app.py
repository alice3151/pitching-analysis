import os
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

st.title("投球フォーム 2画面骨格解析アプリ")
st.write("動画をアップロードすると、解析後に滑らかな動画として生成・再生します。")

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

    # クラウド環境向けの処理サイズ設定 (1画面あたり幅 480px に圧縮して高速化)
    target_w = 480
    target_h = int(height * (target_w / width))
    # 左右2画面並べるため、出力幅は2倍
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

    # Poseモデルの呼び出し
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

            # リサイズ
            frame_resized = cv2.resize(frame, (target_w, target_h))

            # 黒背景画像の作成（右画面用）
            black_bg = np.zeros_like(frame_resized)

            # BGR -> RGB
            image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            results = pose.process(image_rgb)

            # 描画用に複製
            frame_drawn = frame_resized.copy()

            if results.pose_landmarks:
                # 左画面：実映像上に骨格描画
                mp_drawing.draw_landmarks(
                    frame_drawn,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # 右画面：黒背景上に骨格描画 (ワイヤーフレーム風)
                mp_drawing.draw_landmarks(
                    black_bg,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3), # 黄色関節
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # 赤骨格
                )

            # 左右に並べて結合
            combined_frame = np.hstack((frame_drawn, black_bg))

            # 動画ファイルに書き込み
            out.write(combined_frame)

            # 進捗バーの更新
            current_frame += 1
            if total_frames > 0:
                progress_bar.progress(min(current_frame / total_frames, 1.0))

    cap.release()
    out.release()

    status_text.text("解析完了！再生中...")
    progress_bar.empty()

    # 生成した動画を表示
    st.video(output_path)    target_w = 480
    target_h = int(height * (target_w / width))
    # 左右2画面並べるため、出力幅は2倍
    out_w = target_w * 2
    out_h = target_h

    # 一時出力動画ファイルの設定 (mp4v コーデック)
    output_path = os.path.join(tempfile.gettempdir(), "analyzed_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("動画を解析・生成中...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

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

            # 黒背景画像の作成（右画面用）
            black_bg = np.zeros_like(frame_resized)

            # BGR -> RGB
            image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            results = pose.process(image_rgb)

            # 描画用に復元
            frame_drawn = frame_resized.copy()

            if results.pose_landmarks:
                # 左画面：実映像上に骨格描画
                mp_drawing.draw_landmarks(
                    frame_drawn,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )

                # 右画面：黒背景上に骨格描画 (ワイヤーフレーム風)
                mp_drawing.draw_landmarks(
                    black_bg,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 255), thickness=2, circle_radius=3
                    ),  # 黄色関節
                    mp_drawing.DrawingSpec(
                        color=(0, 0, 255), thickness=2
                    ),  # 赤骨格
                )

            # 左右に並べて結合
            combined_frame = np.hstack((frame_drawn, black_bg))

            # 動画ファイルに書き込み
            out.write(combined_frame)

            # 進捗バーの更新
            current_frame += 1
            if total_frames > 0:
                progress_bar.progress(min(current_frame / total_frames, 1.0))

    cap.release()
    out.release()

    status_text.text("解析完了！再生中...")
    progress_bar.empty()

    # 生成した動画を画面に表示 (Streamlit 標準動画プレイヤーなのでカクつきません)
    st.video(output_path)        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("動画の再生・解析が完了しました。")
                break

            # 1. 処理速度向上・メモリ対策のためリサイズ (幅640px)
            height, width = frame.shape[:2]
            target_width = 640
            target_height = int(height * (target_width / width))
            frame_resized = cv2.resize(frame, (target_width, target_height))

            # 2. BGRからRGBに変換
            image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # 3. 骨格検出の実行
            results = pose.process(image_rgb)

            # 4. 描画用に書き込み許可を戻す
            image_rgb.flags.writeable = True

            # 5. 骨格を描画
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_rgb,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

            # 6. Streamlit上に表示
            st_frame.image(image_rgb, channels="RGB", use_container_width=True)

    # リソースの解放
    cap.release()
