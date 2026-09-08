import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import mediapipe as mp


# =========================================================
# MediaPipe
# =========================================================

mp_pose = mp.solutions.pose


# =========================================================
# Streamlit設定
# =========================================================

st.set_page_config(
    page_title="PITCHING KINETIC & ROTATIONAL ANALYSIS",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ ピッチング動作・運動力学解析")

st.sidebar.header("⚙️ 解析設定")


# =========================================================
# ユーザー設定
# =========================================================

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
    "通常撮影 (30 fps)": 30.0,
    "スロー撮影 (60 fps)": 60.0,
    "ハイスピード (120 fps)": 120.0,
    "超スロー (240 fps)": 240.0
}

user_weight = st.sidebar.number_input(
    "体重 (kg)",
    min_value=30.0,
    max_value=120.0,
    value=65.0,
    step=1.0
)

reference_width_m = st.sidebar.number_input(
    "基準股関節幅 (m)",
    min_value=0.12,
    max_value=0.30,
    value=0.18,
    step=0.01,
    help="2D動画のピクセル→m変換に使用します。"
)

smooth_window = st.sidebar.slider(
    "平滑化フレーム数",
    min_value=3,
    max_value=11,
    value=5,
    step=2
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
    単純移動平均。
    NaNにも比較的安全に対応。
    """

    data = np.asarray(data, dtype=float)

    if len(data) == 0:
        return data

    window = max(1, int(window))

    if window == 1:
        return data.copy()

    result = np.full_like(data, np.nan, dtype=float)

    half = window // 2

    for i in range(len(data)):

        start = max(0, i - half)
        end = min(len(data), i + half + 1)

        vals = data[start:end]

        valid = vals[np.isfinite(vals)]

        if len(valid) > 0:
            result[i] = np.mean(valid)

    # 残ったNaNを前後値で補間
    series = pd.Series(result)

    result = (
        series
        .interpolate(limit_direction="both")
        .to_numpy()
    )

    return result


def smooth_points(points, window=5):
    """
    2D座標平滑化
    """

    arr = np.asarray(points, dtype=float)

    if len(arr) == 0:
        return arr

    x = moving_average(
        arr[:, 0],
        window
    )

    y = moving_average(
        arr[:, 1],
        window
    )

    return np.column_stack(
        [x, y]
    )


def distance_2d(p1, p2):
    return float(
        np.linalg.norm(
            np.asarray(p1) -
            np.asarray(p2)
        )
    )


def angle_2d(p1, p2):
    """
    p1→p2 の画像上角度 [deg]
    """

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return float(
        np.degrees(
            np.arctan2(dy, dx)
        )
    )


def unwrap_angle_deg(angle):
    """
    ±180°の境界をまたぐ角度をunwrap
    """

    angle = np.asarray(
        angle,
        dtype=float
    )

    return np.degrees(
        np.unwrap(
            np.radians(angle)
        )
    )


def velocity_1d(position, dt):
    """
    1次元速度
    """

    return np.gradient(
        position,
        dt
    )


def calculate_scale(scales):
    """
    m / pixel
    """

    arr = np.asarray(
        scales,
        dtype=float
    )

    arr = arr[
        np.isfinite(arr)
    ]

    arr = arr[
        arr > 0
    ]

    if len(arr) == 0:
        return 0.001

    return float(
        np.median(arr)
    )


def safe_int_point(point):
    """
    OpenCV用整数座標
    """

    return (
        int(round(point[0])),
        int(round(point[1]))
    )


def find_foot_plant(
    lead_y,
    wrist_speed=None
):
    """
    簡易Foot Plant推定。

    前足Y座標が大きく動き、
    最も下側に近づく領域を候補とする。
    """

    n = len(lead_y)

    if n < 10:
        return max(
            0,
            n // 2
        )

    # 極端な先頭・末尾を除外
    start = int(n * 0.15)
    end = int(n * 0.85)

    if end <= start:
        return n // 2

    candidate = lead_y[
        start:end
    ]

    idx = start + int(
        np.nanargmax(candidate)
    )

    return int(idx)


def calculate_elbow_angle(
    shoulder,
    elbow,
    wrist
):
    """
    肘角度
    """

    v1 = np.asarray(
        shoulder
    ) - np.asarray(elbow)

    v2 = np.asarray(
        wrist
    ) - np.asarray(elbow)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan

    cos_theta = np.dot(
        v1,
        v2
    ) / (
        n1 * n2
    )

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0
    )

    return float(
        np.degrees(
            np.arccos(
                cos_theta
            )
        )
    )


# =========================================================
# アプリ本体
# =========================================================

if uploaded_file is not None:

    # =====================================================
    # 動画保存
    # =====================================================

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    input_file.write(
        uploaded_file.read()
    )

    input_file.close()

    cap = cv2.VideoCapture(
        input_file.name
    )

    if not cap.isOpened():

        st.error(
            "動画を開けませんでした。"
        )

        st.stop()

    orig_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if (
        orig_fps is None
        or not np.isfinite(orig_fps)
        or orig_fps <= 0
    ):
        orig_fps = 30.0

    if video_fps_mode == "動画のFPSを使用":

        fps = orig_fps

    else:

        fps = fps_map[
            video_fps_mode
        ]

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    st.info(
        f"動画: {width} × {height} px / "
        f"元FPS: {orig_fps:.2f} / "
        f"解析FPS: {fps:.2f}"
    )


    # =====================================================
    # 出力動画
    # =====================================================

    overlay_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    overlay_file.close()

    skeleton_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    skeleton_file.close()

    out_overlay_path = (
        overlay_file.name
    )

    out_skeleton_path = (
        skeleton_file.name
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

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


    # =====================================================
    # MediaPipe Landmark
    # =====================================================

    is_right = (
        dominant_hand == "右投げ"
    )

    throwing_shoulder_idx = (
        mp_pose.PoseLandmark.RIGHT_SHOULDER
        if is_right
        else mp_pose.PoseLandmark.LEFT_SHOULDER
    )

    throwing_elbow_idx = (
        mp_pose.PoseLandmark.RIGHT_ELBOW
        if is_right
        else mp_pose.PoseLandmark.LEFT_ELBOW
    )

    throwing_wrist_idx = (
        mp_pose.PoseLandmark.RIGHT_WRIST
        if is_right
        else mp_pose.PoseLandmark.LEFT_WRIST
    )

    pivot_ankle_idx = (
        mp_pose.PoseLandmark.RIGHT_ANKLE
        if is_right
        else mp_pose.PoseLandmark.LEFT_ANKLE
    )

    lead_ankle_idx = (
        mp_pose.PoseLandmark.LEFT_ANKLE
        if is_right
        else mp_pose.PoseLandmark.RIGHT_ANKLE
    )


    # =====================================================
    # データ格納
    # =====================================================

    frames = []

    left_hips = []
    right_hips = []

    left_shoulders = []
    right_shoulders = []

    pelvis_centers = []
    thorax_centers = []

    throwing_shoulders = []
    throwing_elbows = []
    throwing_wrists = []

    pivot_ankles = []
    lead_ankles = []

    scales = []


    # =====================================================
    # PASS 1
    # 骨格検出
    # =====================================================

    st.subheader(
        "① 骨格解析"
    )

    progress = st.progress(0)

    frame_idx = 0

    last_values = None


    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frames.append(
                frame.copy()
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = pose.process(
                rgb
            )

            if results.pose_landmarks:

                lm = results.pose_landmarks.landmark


                # -----------------------------------------
                # Hip
                # -----------------------------------------

                lh = (
                    lm[
                        mp_pose.PoseLandmark.LEFT_HIP
                    ].x * width,

                    lm[
                        mp_pose.PoseLandmark.LEFT_HIP
                    ].y * height
                )

                rh = (
                    lm[
                        mp_pose.PoseLandmark.RIGHT_HIP
                    ].x * width,

                    lm[
                        mp_pose.PoseLandmark.RIGHT_HIP
                    ].y * height
                )

                pelvis = (
                    (lh[0] + rh[0]) / 2.0,
                    (lh[1] + rh[1]) / 2.0
                )


                # -----------------------------------------
                # Shoulder
                # -----------------------------------------

                ls = (
                    lm[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].x * width,

                    lm[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].y * height
                )

                rs = (
                    lm[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER
                    ].x * width,

                    lm[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER
                    ].y * height
                )

                thorax = (
                    (ls[0] + rs[0]) / 2.0,
                    (ls[1] + rs[1]) / 2.0
                )


                # -----------------------------------------
                # Throwing Arm
                # -----------------------------------------

                ts = (
                    lm[
                        throwing_shoulder_idx
                    ].x * width,

                    lm[
                        throwing_shoulder_idx
                    ].y * height
                )

                te = (
                    lm[
                        throwing_elbow_idx
                    ].x * width,

                    lm[
                        throwing_elbow_idx
                    ].y * height
                )

                tw = (
                    lm[
                        throwing_wrist_idx
                    ].x * width,

                    lm[
                        throwing_wrist_idx
                    ].y * height
                )


                # -----------------------------------------
                # Feet
                # -----------------------------------------

                pa = (
                    lm[
                        pivot_ankle_idx
                    ].x * width,

                    lm[
                        pivot_ankle_idx
                    ].y * height
                )

                la = (
                    lm[
                        lead_ankle_idx
                    ].x * width,

                    lm[
                        lead_ankle_idx
                    ].y * height
                )


                # -----------------------------------------
                # Save
                # -----------------------------------------

                left_hips.append(lh)
                right_hips.append(rh)

                left_shoulders.append(ls)
                right_shoulders.append(rs)

                pelvis_centers.append(
                    pelvis
                )

                thorax_centers.append(
                    thorax
                )

                throwing_shoulders.append(
                    ts
                )

                throwing_elbows.append(
                    te
                )

                throwing_wrists.append(
                    tw
                )

                pivot_ankles.append(
                    pa
                )

                lead_ankles.append(
                    la
                )


                # -----------------------------------------
                # Scale
                # -----------------------------------------

                hip_width_px = distance_2d(
                    lh,
                    rh
                )

                if hip_width_px > 5:

                    scales.append(
                        reference_width_m /
                        hip_width_px
                    )


                last_values = (
                    lh,
                    rh,
                    pelvis,
                    ls,
                    rs,
                    thorax,
                    ts,
                    te,
                    tw,
                    pa,
                    la
                )

            else:

                # -----------------------------------------
                # Detection failure
                # -----------------------------------------

                if last_values is not None:

                    (
                        lh,
                        rh,
                        pelvis,
                        ls,
                        rs,
                        thorax,
                        ts,
                        te,
                        tw,
                        pa,
                        la
                    ) = last_values

                else:

                    center = (
                        width / 2.0,
                        height / 2.0
                    )

                    lh = center
                    rh = center
                    pelvis = center

                    ls = center
                    rs = center
                    thorax = center

                    ts = center
                    te = center
                    tw = center

                    pa = center
                    la = center

                left_hips.append(lh)
                right_hips.append(rh)

                left_shoulders.append(ls)
                right_shoulders.append(rs)

                pelvis_centers.append(
                    pelvis
                )

                thorax_centers.append(
                    thorax
                )

                throwing_shoulders.append(
                    ts
                )

                throwing_elbows.append(
                    te
                )

                throwing_wrists.append(
                    tw
                )

                pivot_ankles.append(
                    pa
                )

                lead_ankles.append(
                    la
                )


            frame_idx += 1

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_idx / total_frames,
                        1.0
                    )
                )


    cap.release()


    # =====================================================
    # Validation
    # =====================================================

    num_frames = len(
        frames
    )

    if num_frames < 10:

        st.error(
            "十分なフレーム数を取得できませんでした。"
        )

        st.stop()


    # =====================================================
    # PASS 2
    # 平滑化
    # =====================================================

    st.subheader(
        "② 座標平滑化"
    )

    pelvis_centers = smooth_points(
        pelvis_centers,
        smooth_window
    )

    thorax_centers = smooth_points(
        thorax_centers,
        smooth_window
    )

    left_hips = smooth_points(
        left_hips,
        smooth_window
    )

    right_hips = smooth_points(
        right_hips,
        smooth_window
    )

    left_shoulders = smooth_points(
        left_shoulders,
        smooth_window
    )

    right_shoulders = smooth_points(
        right_shoulders,
        smooth_window
    )

    throwing_shoulders = smooth_points(
        throwing_shoulders,
        smooth_window
    )

    throwing_elbows = smooth_points(
        throwing_elbows,
        smooth_window
    )

    throwing_wrists = smooth_points(
        throwing_wrists,
        smooth_window
    )

    pivot_ankles = smooth_points(
        pivot_ankles,
        smooth_window
    )

    lead_ankles = smooth_points(
        lead_ankles,
        smooth_window
    )

    scale = calculate_scale(
        scales
    )

    dt = 1.0 / fps


    # =====================================================
    # PASS 3
    # 並進・回旋
    # =====================================================

    # -----------------------------------------------------
    # 投球方向
    # -----------------------------------------------------

    stance_vector = np.nanmedian(
        lead_ankles[:, 0] -
        pivot_ankles[:, 0]
    )

    if stance_vector >= 0:

        direction_sign = 1.0

    else:

        direction_sign = -1.0


    # -----------------------------------------------------
    # 骨盤並進
    # -----------------------------------------------------

    pelvis_x_m = (
        pelvis_centers[:, 0] *
        scale
    )

    thorax_x_m = (
        thorax_centers[:, 0] *
        scale
    )

    pelvis_translation = (
        pelvis_x_m *
        direction_sign
    )

    thorax_translation = (
        thorax_x_m *
        direction_sign
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
        smooth_window
    )

    thorax_velocity = moving_average(
        thorax_velocity,
        smooth_window
    )


    # -----------------------------------------------------
    # 骨盤2D速度
    # -----------------------------------------------------

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

    pelvis_speed_2d = moving_average(
        pelvis_speed_2d,
        smooth_window
    )


    # -----------------------------------------------------
    # 骨盤回旋
    # -----------------------------------------------------

    pelvis_angles = np.array(
        [
            angle_2d(
                lh,
                rh
            )
            for lh, rh in zip(
                left_hips,
                right_hips
            )
        ]
    )

    pelvis_angles = unwrap_angle_deg(
        pelvis_angles
    )

    pelvis_rotation_velocity = np.gradient(
        pelvis_angles,
        dt
    )

    pelvis_rotation_velocity = moving_average(
        pelvis_rotation_velocity,
        smooth_window
    )


    # -----------------------------------------------------
    # 胸郭回旋
    # -----------------------------------------------------

    thorax_angles = np.array(
        [
            angle_2d(
                ls,
                rs
            )
            for ls, rs in zip(
                left_shoulders,
                right_shoulders
            )
        ]
    )

    thorax_angles = unwrap_angle_deg(
        thorax_angles
    )

    thorax_rotation_velocity = np.gradient(
        thorax_angles,
        dt
    )

    thorax_rotation_velocity = moving_average(
        thorax_rotation_velocity,
        smooth_window
    )


    # -----------------------------------------------------
    # Separation
    # -----------------------------------------------------

    trunk_separation = (
        thorax_angles -
        pelvis_angles
    )

    trunk_separation = unwrap_angle_deg(
        trunk_separation
    )


    # =====================================================
    # Foot Plant
    # =====================================================

    lead_y = lead_ankles[:, 1]

    foot_plant_idx = find_foot_plant(
        lead_y
    )


    # =====================================================
    # 手首速度
    # =====================================================

    wrist_x_m = (
        throwing_wrists[:, 0] *
        scale
    )

    wrist_y_m = (
        throwing_wrists[:, 1] *
        scale
    )

    wrist_vx = velocity_1d(
        wrist_x_m,
        dt
    )

    wrist_vy = velocity_1d(
        wrist_y_m,
        dt
    )

    wrist_speed = np.sqrt(
        wrist_vx ** 2 +
        wrist_vy ** 2
    )

    wrist_speed = moving_average(
        wrist_speed,
        smooth_window
    )


    # =====================================================
    # Release推定
    # =====================================================

    release_start = min(
        foot_plant_idx + 1,
        num_frames - 1
    )

    release_candidates = wrist_speed[
        release_start:
    ]

    if len(release_candidates) > 0:

        release_idx = (
            release_start +
            int(
                np.nanargmax(
                    release_candidates
                )
            )
        )

    else:

        release_idx = (
            num_frames - 1
        )


    # =====================================================
    # MER
    # =====================================================

    elbow_angles = np.array(
        [
            calculate_elbow_angle(
                s,
                e,
                w
            )
            for s, e, w in zip(
                throwing_shoulders,
                throwing_elbows,
                throwing_wrists
            )
        ]
    )

    elbow_angles = moving_average(
        elbow_angles,
        smooth_window
    )

    mer_start = min(
        foot_plant_idx,
        num_frames - 1
    )

    mer_end = min(
        max(
            release_idx,
            mer_start + 1
        ),
        num_frames - 1
    )

    if mer_end >= mer_start:

        mer_values = elbow_angles[
            mer_start:
            mer_end + 1
        ]

        if np.any(
            np.isfinite(
                mer_values
            )
        ):

            mer_idx = (
                mer_start +
                int(
                    np.nanargmax(
                        mer_values
                    )
                )
            )

        else:

            mer_idx = mer_start

    else:

        mer_idx = mer_start

    mer_angle = float(
        elbow_angles[mer_idx]
    )


    # =====================================================
    # Step Width
    # =====================================================

    step_width_px = distance_2d(
        lead_ankles[foot_plant_idx],
        pivot_ankles[foot_plant_idx]
    )

    step_width_m = (
        step_width_px *
        scale
    )


    # =====================================================
    # 擬似GRF
    # =====================================================

    pelvis_y_m = (
        pelvis_centers[:, 1] *
        scale
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
        smooth_window
    )

    # 画像座標Yの下方向が+
    vertical_acceleration_up = (
        -pelvis_ay
    )

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

    pseudo_grf = moving_average(
        pseudo_grf,
        smooth_window
    )

    pseudo_grf_bw = (
        pseudo_grf /
        (
            user_weight *
            GRAVITY
        )
    )


    # =====================================================
    # 最大値
    # =====================================================

    max_pelvis_velocity = float(
        np.nanmax(
            np.abs(
                pelvis_velocity
            )
        )
    )

    max_thorax_velocity = float(
        np.nanmax(
            np.abs(
                thorax_velocity
            )
        )
    )

    max_pelvis_rotation = float(
        np.nanmax(
            np.abs(
                pelvis_rotation_velocity
            )
        )
    )

    max_thorax_rotation = float(
        np.nanmax(
            np.abs(
                thorax_rotation_velocity
            )
        )
    )

    max_pseudo_grf = float(
        np.nanmax(
            pseudo_grf
        )
    )

    max_pseudo_grf_bw = float(
        np.nanmax(
            pseudo_grf_bw
        )
    )

    max_wrist_speed = float(
        np.nanmax(
            wrist_speed
        )
    )


    # =====================================================
    # Time
    # =====================================================

    times = (
        np.arange(
            num_frames
        ) * dt
    )


    # =====================================================
    # DataFrame
    # =====================================================

    df = pd.DataFrame({

        "Time_s":
        times,

        "Pelvis_Translation_m":
        pelvis_translation,

        "Thorax_Translation_m":
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

        "Wrist_Speed_m_s":
        wrist_speed,

        "Elbow_Angle_2D_deg":
        elbow_angles,

        "Pseudo_GRF_N":
        pseudo_grf,

        "Pseudo_GRF_BW":
        pseudo_grf_bw

    })


    # =====================================================
    # PASS 4
    # 解析動画
    # =====================================================

    st.subheader(
        "③ 解析動画生成"
    )

    video_progress = st.progress(
        0
    )

    # 手首軌道を過去フレーム分保持
    wrist_trail = []

    for i, frame in enumerate(
        frames
    ):

        draw_frame = frame.copy()

        black_frame = np.zeros_like(
            frame
        )


        # =================================================
        # 現在座標
        # =================================================

        pelvis_pt = safe_int_point(
            pelvis_centers[i]
        )

        thorax_pt = safe_int_point(
            thorax_centers[i]
        )

        ts_pt = safe_int_point(
            throwing_shoulders[i]
        )

        te_pt = safe_int_point(
            throwing_elbows[i]
        )

        tw_pt = safe_int_point(
            throwing_wrists[i]
        )

        pa_pt = safe_int_point(
            pivot_ankles[i]
        )

        la_pt = safe_int_point(
            lead_ankles[i]
        )


        # =================================================
        # 投球腕
        # =================================================

        cv2.line(
            draw_frame,
            ts_pt,
            te_pt,
            (0, 255, 255),
            4
        )

        cv2.line(
            draw_frame,
            te_pt,
            tw_pt,
            (0, 255, 255),
            4
        )

        cv2.line(
            black_frame,
            ts_pt,
            te_pt,
            (0, 255, 255),
            4
        )

        cv2.line(
            black_frame,
            te_pt,
            tw_pt,
            (0, 255, 255),
            4
        )


        # =================================================
        # 骨盤
        # =================================================

        cv2.circle(
            draw_frame,
            pelvis_pt,
            11,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            black_frame,
            pelvis_pt,
            11,
            (255, 0, 0),
            -1
        )


        # =================================================
        # 胸郭
        # =================================================

        cv2.circle(
            draw_frame,
            thorax_pt,
            11,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            black_frame,
            thorax_pt,
            11,
            (0, 255, 0),
            -1
        )


        # =================================================
        # 足
        # =================================================

        cv2.circle(
            draw_frame,
            pa_pt,
            7,
            (255, 255, 0),
            -1
        )

        cv2.circle(
            draw_frame,
            la_pt,
            7,
            (255, 255, 0),
            -1
        )


        # =================================================
        # 手首軌道
        # =================================================

        wrist_trail.append(
            tw_pt
        )

        if len(wrist_trail) > 0:

            for k in range(
                1,
                len(wrist_trail)
            ):

                cv2.line(
                    draw_frame,
                    wrist_trail[k - 1],
                    wrist_trail[k],
                    (0, 0, 255),
                    3
                )

                cv2.line(
                    black_frame,
                    wrist_trail[k - 1],
                    wrist_trail[k],
                    (0, 0, 255),
                    3
                )


        # =================================================
        # 骨盤軌道
        # =================================================

        if i > 0:

            prev_pelvis = safe_int_point(
                pelvis_centers[i - 1]
            )

            cv2.line(
                draw_frame,
                prev_pelvis,
                pelvis_pt,
                (255, 0, 0),
                3
            )

            cv2.line(
                black_frame,
                prev_pelvis,
                pelvis_pt,
                (255, 0, 0),
                3
            )


        # =================================================
        # Foot Plant表示
        # =================================================

        if i == foot_plant_idx:

            cv2.line(
                draw_frame,
                pa_pt,
                la_pt,
                (255, 0, 255),
                5
            )

            cv2.putText(
                draw_frame,
                "FOOT PLANT",
                (
                    la_pt[0] + 10,
                    la_pt[1] - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )


        # =================================================
        # MER表示
        # =================================================

        if i == mer_idx:

            cv2.putText(
                draw_frame,
                "MER",
                (
                    tw_pt[0] + 10,
                    tw_pt[1] - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


        # =================================================
        # Release表示
        # =================================================

        if i == release_idx:

            cv2.putText(
                draw_frame,
                "RELEASE",
                (
                    tw_pt[0] + 10,
                    tw_pt[1] + 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # =================================================
        # 画面情報
        # =================================================

        info_lines = [

            f"Time       : {times[i]:.3f} s",

            f"Pelvis Vel : "
            f"{pelvis_velocity[i]:.2f} m/s",

            f"Thorax Vel : "
            f"{thorax_velocity[i]:.2f} m/s",

            f"Pelvis Rot : "
            f"{pelvis_rotation_velocity[i]:.0f} deg/s",

            f"Thorax Rot : "
            f"{thorax_rotation_velocity[i]:.0f} deg/s",

            f"Wrist Vel  : "
            f"{wrist_speed[i]:.2f} m/s",

            f"Pseudo GRF : "
            f"{pseudo_grf[i]:.0f} N"

        ]

        for j, text in enumerate(
            info_lines
        ):

            cv2.putText(
                draw_frame,
                text,
                (
                    30,
                    35 + j * 28
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )


        # =================================================
        # 出力
        # =================================================

        out_overlay.write(
            draw_frame
        )

        out_skeleton.write(
            black_frame
        )

        if i % 5 == 0:

            video_progress.progress(
                min(
                    (i + 1) /
                    num_frames,
                    1.0
                )
            )


    out_overlay.release()
    out_skeleton.release()


    # =====================================================
    # 完了
    # =====================================================

    st.success(
        "解析が完了しました！"
    )


    # =====================================================
    # 指標
    # =====================================================

    st.subheader(
        "📊 ピッチング指標"
    )

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


    # =====================================================
    # イベント
    # =====================================================

    st.subheader(
        "📍 イベント"
    )

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
        "Release",
        f"{times[release_idx]:.3f} s"
    )


    # =====================================================
    # 注意事項
    # =====================================================

    st.warning(
        "この解析は2D動画からの推定です。"
        "骨盤・胸郭回旋速度は画像面内の回旋、"
        "MERは2Dの肘角度を利用したProxy、"
        "GRFは実測値ではなく骨盤鉛直加速度から算出したPseudo GRFです。"
    )


    # =====================================================
    # 動画表示
    # =====================================================

    st.subheader(
        "📹 解析動画"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "実動画 + 解析"
        )

        st.video(
            out_overlay_path
        )

    with col2:

        st.write(
            "軌道・骨格"
        )

        st.video(
            out_skeleton_path
        )


    # =====================================================
    # Graph 1
    # 並進
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
        x=times[
            foot_plant_idx
        ],
        line_dash="dash",
        annotation_text="Foot Plant"
    )

    fig1.add_vline(
        x=times[
            mer_idx
        ],
        line_dash="dash",
        annotation_text="MER"
    )

    fig1.add_vline(
        x=times[
            release_idx
        ],
        line_dash="dot",
        annotation_text="Release"
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
    # 回旋
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
        x=times[
            foot_plant_idx
        ],
        line_dash="dash",
        annotation_text="Foot Plant"
    )

    fig2.add_vline(
        x=times[
            mer_idx
        ],
        line_dash="dash",
        annotation_text="MER"
    )

    fig2.add_vline(
        x=times[
            release_idx
        ],
        line_dash="dot",
        annotation_text="Release"
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
        x=times[
            foot_plant_idx
        ],
        line_dash="dash",
        annotation_text="Foot Plant"
    )

    fig3.add_vline(
        x=times[
            mer_idx
        ],
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
        "🦾 投球腕 2D角度"
    )

    fig4 = go.Figure()

    fig4.add_trace(
        go.Scatter(
            x=times,
            y=elbow_angles,
            mode="lines",
            name="Elbow Angle"
        )
    )

    fig4.add_trace(
        go.Scatter(
            x=[
                times[mer_idx]
            ],
            y=[
                elbow_angles[mer_idx]
            ],
            mode="markers+text",
            text=["MER"],
            textposition="top center",
            name="MER"
        )
    )

    fig4.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Elbow Angle (deg)",
        height=350,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )


    # =====================================================
    # Graph 5
    # 手首速度
    # =====================================================

    st.subheader(
        "🖐️ 手首速度"
    )

    fig5 = go.Figure()

    fig5.add_trace(
        go.Scatter(
            x=times,
            y=wrist_speed,
            mode="lines",
            name="Wrist Speed"
        )
    )

    fig5.add_vline(
        x=times[
            foot_plant_idx
        ],
        line_dash="dash",
        annotation_text="Foot Plant"
    )

    fig5.add_vline(
        x=times[
            release_idx
        ],
        line_dash="dot",
        annotation_text="Release"
    )

    fig5.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Wrist Speed (m/s)",
        height=350,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )


    # =====================================================
    # Graph 6
    # 擬似GRF
    # =====================================================

    st.subheader(
        "🦶 擬似地面反力"
    )

    fig6 = go.Figure()

    fig6.add_trace(
        go.Scatter(
            x=times,
            y=pseudo_grf,
            mode="lines",
            name="Pseudo GRF"
        )
    )

    fig6.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Force (N)",
        height=350,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )


    # =====================================================
    # CSV
    # =====================================================

    st.subheader(
        "📥 解析データ"
    )

    csv_data = df.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )

    st.download_button(
        label="CSVをダウンロード",
        data=csv_data,
        file_name="pitching_analysis.csv",
        mime="text/csv"
    )
