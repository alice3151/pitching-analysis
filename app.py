import tempfile
import cv2
import mediapipe as mp
import streamlit as st

# ページ基本設定
st.title("投球フォーム 骨格解析アプリ")
st.write("動画ファイルをアップロードして、投球フォームの骨格を検出します。")

# MediaPipe の初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 動画ファイルのアップローダー
uploaded_file = st.file_uploader(
    "動画を選択してください (mp4, mov, avi)", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
    # 拡張子を維持して一時保存ファイルを作成
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.read())
        temp_filename = tfile.name

    # OpenCVで動画をオープン
    cap = cv2.VideoCapture(temp_filename)

    # 画面表示用のStreamlit要素
    st_frame = st.empty()

    # Poseモデルの呼び出し
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
