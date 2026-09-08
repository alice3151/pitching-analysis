import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# =========================================================
# 設定
# =========================================================

st.set_page_config(
    page_title="PITCHING KINETIC & ROTATIONAL ANALYSIS",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ ピッチング動作・運動力学解析")

st.sidebar.header("⚙️ 解析設定")

dominant_hand = st.sidebar.radio(
    "投手タイプ",
    ["右投げ", "左投げ"]
)

video_fps_mode = st.sidebar.selectbox(
    "撮影スピード設定",
    [
        "動画のFPSを使用",
        "通常撮影 (30 fps)",
        "スロー撮影 (60 fps)",
        "ハイスピード (120 fps)",
        "超スロー (240 fps)"
    ]
)

fps_map = {
    "通常撮影 (30 fps)": 30,
    "スロー撮影 (60 fps)": 60,
    "ハイスピード (120 fps)": 120,
    "超スロー (240 fps)": 240
}

user_weight = st.sidebar.number_input(
    "体重 (kg)",
    min_value=30.0,
    max_value=120.0,
    value=65.0,
    step=1.0
)

# 股関節幅を基準にする場合
reference_width_m = st.sidebar.number_input(
    "基準となる股関節幅 (m)",
    min_value=0.12,
    max_value=0.30,
    value=0.18,
    step=0.01
)

GRAVITY = 9.81

uploaded_file = st.file_uploader(
    "動画ファイルをアップロードしてください",
    type=["mp4", "mov", "avi"]
)


# =========================================================
# Utility
# =========================================================

def moving_average(data, window=5):
    """
    左寄り移動平均ではなく、中央寄りの単純移動平均。
    """
    if len(data) == 0:
        return data

    data = np.asarray(data, dtype=float)

    kernel = np.ones(window) / window
    result = np.convolve(data, kernel, mode="same")

    # 端点を補正
    half = window // 2

    for i in range(half):
        result[i] = np.mean(data[:i + half + 1])

    for i in range(len(data) - half, len(data)):
        result[i] = np.mean(data[i - half:])

    return result


def smooth_points(points, window=5):
    """
    2D座標の平滑化
    """
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return points

    x = moving_average(points[:, 0], window)
    y = moving_average(points[:, 1], window)

    return np.column_stack([x, y])


def angle_2d(p1, p2):
    """
    p1 -> p2 の角度 [deg]
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return np.degrees(np.arctan2(dy, dx))


def unwrap_angle_deg(angle):
    """
    -180~180の角度をunwrap
    """
    rad = np.radians(angle)
    return np.degrees(np.unwrap(rad))


def calculate_angular_velocity(angle_deg, dt):
    """
    deg/s
    """
    return np.gradient(angle_deg, dt)


def distance_2d(p1, p2):
    return np.linalg.norm(np.asarray(p1) - np.asarray(p2))


def velocity_1d(position, dt):
    return np.gradient(position, dt)


def find_foot_plant(lead_y, fps):
    """
    前足接地を簡易推定。
    画面座標では下方向が+なので、
    前足が最も下側に来る付近を候補とする。

    ※完全自動のFC検出ではない
    """
    n = len(lead_y)

    if n < 10:
        return max(0, n // 2)

    start = int(n * 0.20)
    end = int(n * 0.85)

    # 最も下に来た位置
    idx = start + np.argmax(lead_y[start:end])

    return int(idx)


def calculate_scale(scales):
    """
    m / px
    """
    scales = np.asarray(scales)

    scales = scales[np.isfinite(scales)]
    scales = scales[scales > 0]

    if len(scales) == 0:
        return 0.001

    return float(np.median(scales))


# =========================================================
# Main
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # 一時ファイル
    # -----------------------------------------------------

    suffix = os.path.splitext(uploaded_file.name)[1]

    tfile = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)

    orig_fps = cap.get(cv2.CAP_PROP_FPS)

    if orig_fps is None or orig_fps <= 0:
        orig_fps = 30.0

    if video_fps_mode == "動画のFPSを使用":
        fps = orig_fps
    else:
        fps = fps_map[video_fps_mode]

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.write(
        f"動画サイズ: {width} × {height} px / "
        f"FPS: {orig_fps:.2f}"
    )

    # -----------------------------------------------------
    # 出力動画
    # -----------------------------------------------------

    out_overlay_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    out_skeleton_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_overlay = cv2.VideoWriter(
        out_overlay_path,
        fourcc,
        orig_fps,
        (width, height)
    )

    out_skeleton = cv2.VideoWriter(
        out_skeleton_path,
        fourcc,
        orig_fps,
        (width, height)
    )

    progress_bar = st.progress(0)

    # =====================================================
    # Landmark index
    # =====================================================

    is_right = dominant_hand == "右投げ"

    throwing_shoulder = (
        mp_pose.PoseLandmark.RIGHT_SHOULDER
        if is_right
        else mp_pose.PoseLandmark.LEFT_SHOULDER
    )

    throwing_elbow = (
        mp_pose.PoseLandmark.RIGHT_ELBOW
        if is_right
        else mp_pose.PoseLandmark.LEFT_ELBOW
    )

    throwing_wrist = (
        mp_pose.PoseLandmark.RIGHT_WRIST
        if is_right
        else mp_pose.PoseLandmark.LEFT_WRIST
    )

    pivot_ankle = (
        mp_pose.PoseLandmark.RIGHT_ANKLE
        if is_right
        else mp_pose.PoseLandmark.LEFT_ANKLE
    )

    lead_ankle = (
        mp_pose.PoseLandmark.LEFT_ANKLE
        if is_right
        else mp_pose.PoseLandmark.RIGHT_ANKLE
    )

    # =====================================================
    # Data Buffer
    # =====================================================

    frames = []

    pelvis_centers = []
    thorax_centers = []

    left_hips = []
    right_hips = []

    left_shoulders = []
    right_shoulders = []

    throwing_shoulders = []
    throwing_elbows = []
    throwing_wrists = []

    pivot_ankles = []
    lead_ankles = []

    scales = []

    # =====================================================
    # PASS 1
    # Pose Detection
    # =====================================================

    st.info("① 骨格解析中...")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        frame_idx = 0

        last_values = None

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame.copy())

            image_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = pose.process(image_rgb)

            if results.pose_landmarks:

                lm = results.pose_landmarks.landmark

                # -----------------------------------------
                # Hip
                # -----------------------------------------

                lh = (
                    lm[mp_pose.PoseLandmark.LEFT_HIP].x * width,
                    lm[mp_pose.PoseLandmark.LEFT_HIP].y * height
                )

                rh = (
                    lm[mp_pose.PoseLandmark.RIGHT_HIP].x * width,
                    lm[mp_pose.PoseLandmark.RIGHT_HIP].y * height
                )

                pelvis = (
                    (lh[0] + rh[0]) / 2,
                    (lh[1] + rh[1]) / 2
                )

                # -----------------------------------------
                # Shoulder
                # -----------------------------------------

                ls = (
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width,
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height
                )

                rs = (
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height
                )

                thorax = (
                    (ls[0] + rs[0]) / 2,
                    (ls[1] + rs[1]) / 2
                )

                # -----------------------------------------
                # Throwing arm
                # -----------------------------------------

                ts = (
                    lm[throwing_shoulder].x * width,
                    lm[throwing_shoulder].y * height
                )

                te = (
                    lm[throwing_elbow].x * width,
                    lm[throwing_elbow].y * height
                )

                tw = (
                    lm[throwing_wrist].x * width,
                    lm[throwing_wrist].y * height
                )

                # -----------------------------------------
                # Feet
                # -----------------------------------------

                pa = (
                    lm[pivot_ankle].x * width,
                    lm[pivot_ankle].y * height
                )

                la = (
                    lm[lead_ankle].x * width,
                    lm[lead_ankle].y * height
                )

                # -----------------------------------------
                # Save
                # -----------------------------------------

                left_hips.append(lh)
                right_hips.append(rh)
                pelvis_centers.append(pelvis)

                left_shoulders.append(ls)
                right_shoulders.append(rs)
                thorax_centers.append(thorax)

                throwing_shoulders.append(ts)
                throwing_elbows.append(te)
                throwing_wrists.append(tw)

                pivot_ankles.append(pa)
                lead_ankles.append(la)

                # -----------------------------------------
                # Scale
                # hip width → meters
                # -----------------------------------------

                hip_width_px = distance_2d(lh, rh)

                if hip_width_px > 5:

                    scales.append(
                        reference_width_m / hip_width_px
                    )

                last_values = (
                    lh, rh, pelvis,
                    ls, rs, thorax,
                    ts, te, tw,
                    pa, la
                )

            else:

                # -----------------------------------------
                # Detection failure
                # 前フレーム保持
                # -----------------------------------------

                if last_values is not None:

                    (
                        lh, rh, pelvis,
                        ls, rs, thorax,
                        ts, te, tw,
                        pa, la
                    ) = last_values

                else:

                    center = (width / 2, height / 2)

                    lh = rh = pelvis = center
                    ls = rs = thorax = center
                    ts = te = tw = center
                    pa = la = center

                left_hips.append(lh)
                right_hips.append(rh)
                pelvis_centers.append(pelvis)

                left_shoulders.append(ls)
                right_shoulders.append(rs)
                thorax_centers.append(thorax)

                throwing_shoulders.append(ts)
                throwing_elbows.append(te)
                throwing_wrists.append(tw)

                pivot_ankles.append(pa)
                lead_ankles.append(la)

            frame_idx += 1

            progress_bar.progress(
                min(frame_idx / max(total_frames, 1), 1.0)
            )

    cap.release()

    num_frames = len(frames)

    if num_frames < 10:
        st.error("動画が短すぎます。")
        st.stop()

    # =====================================================
    # PASS 2
    # Smooth
    # =====================================================

    pelvis_centers = smooth_points(
        pelvis_centers,
        window=7
    )

    thorax_centers = smooth_points(
        thorax_centers,
        window=7
    )

    left_hips = smooth_points(
        left_hips,
        window=7
    )

    right_hips = smooth_points(
        right_hips,
        window=7
    )

    left_shoulders = smooth_points(
        left_shoulders,
        window=7
    )

    right_shoulders = smooth_points(
        right_shoulders,
        window=7
    )

    throwing_shoulders = smooth_points(
        throwing_shoulders,
        window=7
    )

    throwing_elbows = smooth_points(
        throwing_elbows,
        window=7
    )

    throwing_wrists = smooth_points(
        throwing_wrists,
        window=7
    )

    pivot_ankles = smooth_points(
        pivot_ankles,
        window=7
    )

    lead_ankles = smooth_points(
        lead_ankles,
        window=7
    )

    scale = calculate_scale(scales)

    dt = 1.0 / fps

    # =====================================================
    # PASS 3
    # 投球方向軸を決める
    # =====================================================

    # 初期の足幅方向から投球方向を推定
    stance_vector = (
        np.mean(lead_ankles[:, 0] - pivot_ankles[:, 0])
    )

    direction_sign = 1.0 if stance_vector >= 0 else -1.0

    # =====================================================
    # ① 骨盤・胸郭 並進
    # =====================================================

    pelvis_x_m = pelvis_centers[:, 0] * scale
    thorax_x_m = thorax_centers[:, 0] * scale

    # 投球方向を + とする
    pelvis_translation = (
        pelvis_x_m * direction_sign
    )

    thorax_translation = (
        thorax_x_m * direction_sign
    )

    pelvis_velocity = velocity_1d(
        pelvis_translation,
        dt
    )

    thorax_velocity = velocity_1d(
        thorax_translation,
        dt
    )

    pelvis_velocity = moving_average(
        pelvis_velocity,
        5
    )

    thorax_velocity = moving_average(
        thorax_velocity,
        5
    )

    # 2D速度の大きさも保存
    pelvis_vx = velocity_1d(
        pelvis_centers[:, 0] * scale,
        dt
    )

    pelvis_vy = velocity_1d(
        pelvis_centers[:, 1] * scale,
        dt
    )

    pelvis_speed_2d = np.sqrt(
        pelvis_vx ** 2 +
        pelvis_vy ** 2
    )

    # =====================================================
    # ② 骨盤回旋
    # =====================================================

    pelvis_angles = np.array([
        angle_2d(lh, rh)
        for lh, rh
        in zip(left_hips, right_hips)
    ])

    pelvis_angles = unwrap_angle_deg(
        pelvis_angles
    )

    pelvis_rotation_velocity = calculate_angular_velocity(
        pelvis_angles,
        dt
    )

    pelvis_rotation_velocity = moving_average(
        pelvis_rotation_velocity,
        5
    )

    # =====================================================
    # ③ 胸郭回旋
    # =====================================================

    thorax_angles = np.array([
        angle_2d(ls, rs)
        for ls, rs
        in zip(left_shoulders, right_shoulders)
    ])

    thorax_angles = unwrap_angle_deg(
        thorax_angles
    )

    thorax_rotation_velocity = calculate_angular_velocity(
        thorax_angles,
        dt
    )

    thorax_rotation_velocity = moving_average(
        thorax_rotation_velocity,
        5
    )

    # =====================================================
    # ④ 骨盤 → 胸郭の separation
    # =====================================================

    trunk_separation = (
        thorax_angles - pelvis_angles
    )

    trunk_separation = unwrap_angle_deg(
        trunk_separation
    )

    # =====================================================
    # ⑤ 簡易GRF
    # =====================================================

    # -----------------------------------------
    # 骨盤の上下加速度
    # -----------------------------------------

    pelvis_y_m = (
        pelvis_centers[:, 1] * scale
    )

    pelvis_vy_m = velocity_1d(
        pelvis_y_m,
        dt
    )

    pelvis_ay = velocity_1d(
        pelvis_vy_m,
        dt
    )

    pelvis_ay = moving_average(
        pelvis_ay,
        5
    )

    # 画像yは下方向が+
    vertical_acceleration_up = -pelvis_ay

    # -----------------------------------------
    # 擬似GRF
    # -----------------------------------------

    pseudo_grf = (
        user_weight *
        (
            vertical_acceleration_up +
            GRAVITY
        )
    )

    pseudo_grf = np.maximum(
        pseudo_grf,
        0
    )

    # -----------------------------------------
    # 体重比
    # -----------------------------------------

    pseudo_grf_bw = (
        pseudo_grf /
        (user_weight * GRAVITY)
    )

    # =====================================================
    # ⑥ Foot Plant
    # =====================================================

    lead_y = lead_ankles[:, 1]

    foot_plant_idx = find_foot_plant(
        lead_y,
        fps
    )

    # =====================================================
    # ⑦ MER
    # =====================================================

    # 肘角度
    # shoulder -> elbow
    # wrist -> elbow

    elbow_angles = []

    for s, e, w in zip(
        throwing_shoulders,
        throwing_elbows,
        throwing_wrists
    ):

        v1 = s - e
        v2 = w - e

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            elbow_angles.append(np.nan)
            continue

        cos_theta = np.dot(v1, v2) / (
            norm1 * norm2
        )

        cos_theta = np.clip(
            cos_theta,
            -1,
            1
        )

        theta = np.degrees(
            np.arccos(cos_theta)
        )

        elbow_angles.append(theta)

    elbow_angles = np.array(
        elbow_angles
    )

    elbow_angles = moving_average(
        elbow_angles,
        5
    )

    # -----------------------------------------
    # Releaseの簡易推定
    # 手首速度ピーク
    # -----------------------------------------

    wrist_x = (
        throwing_wrists[:, 0] * scale
    )

    wrist_y = (
        throwing_wrists[:, 1] * scale
    )

    wrist_vx = velocity_1d(
        wrist_x,
        dt
    )

    wrist_vy = velocity_1d(
        wrist_y,
        dt
    )

    wrist_speed = np.sqrt(
        wrist_vx ** 2 +
        wrist_vy ** 2
    )

    wrist_speed = moving_average(
        wrist_speed,
        5
    )

    # Plantより後のピーク
    release_start = min(
        foot_plant_idx + 1,
        num_frames - 1
    )

    release_idx = release_start + np.argmax(
        wrist_speed[
            release_start:
        ]
    )

    # MER候補
    # FC ～ Releaseの間で肘角最大
    mer_start = foot_plant_idx
    mer_end = max(
        release_idx,
        mer_start + 1
    )

    if mer_end > mer_start:

        mer_relative_idx = np.argmax(
            elbow_angles[
                mer_start:
                mer_end + 1
            ]
        )

        mer_idx = (
            mer_start +
            mer_relative_idx
        )

    else:

        mer_idx = foot_plant_idx

    mer_angle = elbow_angles[mer_idx]

    # =====================================================
    # ⑧ Step Width
    # =====================================================

    step_width_px = distance_2d(
        lead_ankles[foot_plant_idx],
        pivot_ankles[foot_plant_idx]
    )

    step_width_m = (
        step_width_px * scale
    )

    # =====================================================
    # 数値結果
    # =====================================================

    max_pelvis_velocity = np.max(
        np.abs(pelvis_velocity)
    )

    max_thorax_velocity = np.max(
        np.abs(thorax_velocity)
    )

    max_pelvis_rotation = np.max(
        np.abs(pelvis_rotation_velocity)
    )

    max_thorax_rotation = np.max(
        np.abs(thorax_rotation_velocity)
    )

    max_pseudo_grf = np.max(
        pseudo_grf
    )

    max_pseudo_grf_bw = np.max(
        pseudo_grf_bw
    )

    max_wrist_speed = np.max(
        wrist_speed
    )

    # =====================================================
    # DataFrame
    # =====================================================

    times = np.arange(
        num_frames
    ) * dt

    df = pd.DataFrame({

        "Time_s":
        times,

        "Pelvis_X_m":
        pelvis_translation,

        "Thorax_X_m":
        thorax_translation,

        "Pelvis_Translation_Velocity_m_s":
        pelvis_velocity,

        "Thorax_Translation_Velocity_m_s":
        thorax_velocity,

        "Pelvis_Rotation_deg":
        pelvis_angles,

        "Thorax_Rotation_deg":
        thorax_angles,

        "Pelvis_Rotation_Velocity_deg_s":
        pelvis_rotation_velocity,

        "Thorax_Rotation_Velocity_deg_s":
        thorax_rotation_velocity,

        "Trunk_Separation_deg":
        trunk_separation,

        "Pelvis_Vertical_Acceleration_m_s2":
        vertical_acceleration_up,

        "Pseudo_GRF_N":
        pseudo_grf,

        "Pseudo_GRF_BW":
        pseudo_grf_bw,

        "Wrist_Speed_m_s":
        wrist_speed,

        "Elbow_Angle_deg":
        elbow_angles

    })

    # =====================================================
    # PASS 4
    # 動画描画
    # =====================================================

    st.info("② 解析動画生成中...")

    skeleton_connections = mp_pose.POSE_CONNECTIONS

    for i, frame in enumerate(frames):

        draw_frame = frame.copy()

        black_frame = np.zeros_like(
            frame
        )

        # -------------------------------------------------
        # 骨格
        # -------------------------------------------------

        points = {
            "LH": left_hips[i],
            "RH": right_hips[i],
            "LS": left_shoulders[i],
            "RS": right_shoulders[i],
            "TS": throwing_shoulders[i],
            "TE": throwing_elbows[i],
            "TW": throwing_wrists[i],
            "PA": pivot_ankles[i],
            "LA": lead_ankles[i],
            "P": pelvis_centers[i],
            "T": thorax_centers[i]
        }

        # -------------------------------------------------
        # Pelvis
        # -------------------------------------------------

        p = tuple(
            np.int32(
                np.round(
                    pelvis_centers[i]
                )
            )
        )

        cv2.circle(
            draw_frame,
            p,
            10,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            black_frame,
            p,
            10,
            (255, 0, 0),
            -1
        )

        # -------------------------------------------------
        # Thorax
        # -------------------------------------------------

        t = tuple(
            np.int32(
                np.round(
                    thorax_centers[i]
                )
            )
        )

        cv2.circle(
            draw_frame,
            t,
            10,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            black_frame,
            t,
            10,
            (0, 255, 0),
            -1
        )

        # -------------------------------------------------
        # Throwing Arm
        # -------------------------------------------------

        ts = tuple(
            np.int32(
                np.round(
                    throwing_shoulders[i]
                )
            )
        )

        te = tuple(
            np.int32(
                np.round(
                    throwing_elbows[i]
                )
            )
        )

        tw = tuple(
            np.int32(
                np.round(
                    throwing_wrists[i]
                )
            )
        )

        cv2.line(
            draw_frame,
            ts,
            te,
            (0, 255, 255),
            4
        )

        cv2.line(
            draw_frame,
            te,
            tw,
            (0, 255, 255),
            4
        )

        cv2.line(
            black_frame,
            ts,
            te,
            (0, 255, 255),
            4
        )

        cv2.line(
            black_frame,
            te,
            tw,
            (0, 255, 255),
            4
        )

        # -------------------------------------------------
        # 軌道
        # -------------------------------------------------

        if i > 1:

            prev_p = tuple(
                np.int32(
                    np.round(
                        pelvis_centers[i - 1]
                    )
                )
            )

            cv2.line(
                draw_frame,
                prev_p,
                p,
                (255, 0, 0),
                3
            )

            cv2.line(
                black_frame,
                prev_p,
                p,
                (255, 0, 0),
                3
            )

        # -------------------------------------------------
        # MER frame
        # -------------------------------------------------

        if i == mer_idx:

            cv2.putText(
                draw_frame,
                "MER",
                (
                    int(
                        throwing_wrists[i][0]
                    ) + 10,
                    int(
                        throwing_wrists[i][1]
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        # -------------------------------------------------
        # Foot Plant
        # -------------------------------------------------

        if i == foot_plant_idx:

            cv2.putText(
                draw_frame,
                "FOOT PLANT",
                (
                    int(
                        lead_ankles[i][0]
                    ),
                    int(
                        lead_ankles[i][1] - 20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.line(
                draw_frame,
                tuple(
                    np.int32(
                        np.round(
                            pivot_ankles[i]
                        )
                    )
                ),
                tuple(
                    np.int32(
                        np.round(
                            lead_ankles[i]
                        )
                    )
                ),
                (255, 0, 255),
                4
            )

        # -------------------------------------------------
        # 情報表示
        # -------------------------------------------------

        cv2.putText(
            draw_frame,
            f"Pelvis Vel : {pelvis_velocity[i]:.2f} m/s",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            draw_frame,
            f"Thorax Vel : {thorax_velocity[i]:.2f} m/s",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            draw_frame,
            f"Pelvis Rot : {pelvis_rotation_velocity[i]:.1f} deg/s",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            draw_frame,
            f"Thorax Rot : {thorax_rotation_velocity[i]:.1f} deg/s",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            draw_frame,
            f"MER : {elbow_angles[i]:.1f} deg",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            draw_frame,
            f"Pseudo GRF : {pseudo_grf[i]:.0f} N",
            (30, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        out_overlay.write(
            draw_frame
        )

        out_skeleton.write(
            black_frame
        )

        if i % 5 == 0:

            progress_bar.progress(
                min(
                    (i + 1) /
                    num_frames,
                    1.0
                )
            )

    out_overlay.release()
    out_skeleton.release()

    st.success("解析完了！")

    # =====================================================
    # 数値表示
    # =====================================================

    st.subheader("📊 ピッチング指標")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "骨盤最大並進速度",
        f"{max_pelvis_velocity:.2f} m/s"
    )

    c2.metric(
        "胸郭最大並進速度",
        f"{max_thorax_velocity:.2f} m/s"
    )

    c3.metric(
        "骨盤最大回旋速度",
        f"{max_pelvis_rotation:.0f} °/s"
    )

    c4.metric(
        "胸郭最大回旋速度",
        f"{max_thorax_rotation:.0f} °/s"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "MER 2D推定",
        f"{mer_angle:.1f}°"
    )

    c6.metric(
        "ステップ幅",
        f"{step_width_m:.2f} m"
    )

    c7.metric(
        "最大手速度",
        f"{max_wrist_speed:.2f} m/s"
    )

    c8.metric(
        "最大擬似GRF",
        f"{max_pseudo_grf:.0f} N"
    )

    st.caption(
        "※ GRFは骨盤鉛直加速度から算出した擬似GRFです。"
        "フォースプレート等による実測地面反力ではありません。"
    )

    st.caption(
        "※ MERは2D動画からの外旋角度推定値です。"
        "真の肩関節外旋角度を取得するには3D計測が必要です。"
    )

    # =====================================================
    # Event
    # =====================================================

    st.subheader("📍 イベント")

    e1, e2, e3 = st.columns(3)

    e1.metric(
        "Foot Plant",
        f"{times[foot_plant_idx]:.3f} s"
    )

    e2.metric(
        "MER",
        f"{times[mer_idx]:.3f} s"
    )

    e3.metric(
        "Release推定",
        f"{times[release_idx]:.3f} s"
    )

    # =====================================================
    # 動画
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "📹 実動画 + 解析"
        )

        st.video(
            out_overlay_path
        )

    with col2:

        st.subheader(
            "🦴 骨格・軌道"
        )

        st.video(
            out_skeleton_path
        )

    # =====================================================
    # Graph 1
    # 並進速度
    # =====================================================

    st.subheader(
        "📈 骨盤・胸郭 並進速度"
    )

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x=times,
            y=pelvis_velocity,
            mode="lines",
            name="Pelvis"
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=times,
            y=thorax_velocity,
            mode="lines",
            name="Thorax"
        )
    )

    fig1.add_vline(
        x=times[foot_plant_idx],
        line_dash="dash",
        annotation_text="Foot Plant"
    )

    fig1.add_vline(
        x=times[mer_idx],
        line_dash="dash",
        annotation_text="MER"
    )

    fig1.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Translation Velocity (m/s)",
        height=400,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

  # =====================================================
# Graph 2
# 回旋速度
# =====================================================

st.subheader(
    "🔄 骨盤・胸郭 回旋速度"
)

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=times,
        y=pelvis_rotation_velocity,
        mode="lines",
        name="Pelvis Rotation"
    )
)

fig2.add_trace(
    go.Scatter(
        x=times,
        y=thorax_rotation_velocity,
        mode="lines",
        name="Thorax Rotation"
    )
)

fig2.add_vline(
    x=times[foot_plant_idx],
    line_dash="dash"
)

fig2.add_vline(
    x=times[mer_idx],
    line_dash="dash"
)

fig2.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Angular Velocity (deg/s)",
    height=400,
    template="plotly_dark"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

    # =====================================================
    # Graph 3
    # Separation
    # =====================================================

       # =====================================================
    # Graph 3
    # Separation
    # =====================================================

    st.subheader(
        "↔️ 骨盤−胸郭 Separation"
    )

    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            x=times,
            y=trunk_separation,
            mode="lines",
            name="Pelvis-Thorax"
        )
    )

    fig3.add_vline(
        x=times[foot_plant_idx],
        line_dash="dash",
        annotation_text="Foot Plant"
    )

    fig3.add_vline(
        x=times[mer_idx],
        line_dash="dash",
        annotation_text="MER"
    )

    fig3.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Separation Angle (deg)",
        height=350,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # =====================================================
    # Graph 4
    # MER
    # =====================================================

    st.subheader(
        "🦾 投球腕 2D外旋推定"
    )

    fig4 = go.Figure()

    fig4.add_trace(
        go.Scatter(
            x=times,
            y=elbow_angles,
            mode="lines",
            name="2D MER Proxy"
        )
    )

    fig4.add_trace(
        go.Scatter(
            x=[times[mer_idx]],
            y=[elbow_angles[mer_idx]],
            mode="markers+text",
            text=["MER"],
            textposition="top center",
            name="MER"
        )
    )

    fig4.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Arm Angle (deg)",
        height=350,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # =====================================================
    # Graph 5
    # Pseudo GRF
    # =====================================================

    st.subheader(
        "🦶 擬似地面反力"
    )

    fig5 = go.Figure()

    fig5.add_trace(
        go.Scatter(
            x=times,
            y=pseudo_grf,
            mode="lines",
            name="Pseudo GRF"
        )
    )

    fig5.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Force (N)",
        height=350,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    # =====================================================
    # CSV download
    # =====================================================

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 解析データCSVをダウンロード",
        data=csv_data,
        file_name="pitching_analysis.csv",
        mime="text/csv"
    )
