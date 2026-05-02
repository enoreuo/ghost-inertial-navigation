"""Data Loading for TLIO EKF2 Replacement (TensorFlow version)

Reads IMU and ground-truth data in EuRoC MAV, PX4 SITL, or TLIO golden format.
Returns numpy arrays ready for Keras model.fit().

Supported formats:
  - EuRoC MAV CSV (imu0/data.csv + state_groundtruth_estimate0/data.csv)
  - PX4 SITL CSV (sensor_combined + vehicle_local_position)
  - TLIO golden .npy (imu0_resampled.npy — world-frame IMU rotated to body frame)
"""

import os
import numpy as np

# Normalization constants
ACC_NORM = 9.81       # 1g in m/s^2
GYRO_NORM = 3.14159   # pi rad/s
WINDOW_SIZE = 200     # 1 second at 200Hz
STRIDE = 100          # 50% overlap


def detect_format(filepath):
    """Detect CSV format: 'euroc' or 'px4'.

    Reads the first line to determine column naming convention.
    """
    with open(filepath, 'r') as f:
        header = f.readline().strip()

    if '#timestamp' in header or 'w_RS_S' in header or 'a_RS_S' in header:
        return 'euroc'
    elif 'gyro_rad' in header or 'accelerometer_m_s2' in header:
        return 'px4'
    elif 'timestamp' in header:
        return 'px4'  # generic PX4 ulog format
    else:
        raise ValueError(f"Unknown CSV format in {filepath}. "
                         f"Expected EuRoC or PX4 SITL format.\nHeader: {header}")


def load_imu_csv(filepath):
    """Load IMU CSV and return (timestamps_ns, imu_data).

    Returns:
        timestamps: int64 array of nanosecond timestamps, shape (N,)
        imu_data: float32 array [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], shape (N, 6)
    """
    fmt = detect_format(filepath)

    if fmt == 'euroc':
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        timestamps = data[:, 0].astype(np.int64)
        gyro = data[:, 1:4].astype(np.float32)
        acc = data[:, 4:7].astype(np.float32)
        imu_data = np.concatenate([acc, gyro], axis=1)  # reorder: acc first

    elif fmt == 'px4':
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        timestamps = (data[:, 0] * 1000).astype(np.int64)
        if data.shape[1] >= 7:
            gyro = data[:, 1:4].astype(np.float32)
            acc = data[:, 4:7].astype(np.float32)
            imu_data = np.concatenate([acc, gyro], axis=1)
        else:
            raise ValueError(f"PX4 IMU CSV has {data.shape[1]} columns, expected >= 7")

    return timestamps, imu_data


def load_gt_csv(filepath):
    """Load ground truth CSV and return (timestamps_ns, positions, quaternions).

    Returns:
        timestamps: int64 array of nanosecond timestamps, shape (N,)
        positions: float32 array [x, y, z] in meters, shape (N, 3)
        quaternions: float32 array [q_w, q_x, q_y, q_z], shape (N, 4)
    """
    fmt = detect_format(filepath)

    if fmt == 'euroc':
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        timestamps = data[:, 0].astype(np.int64)
        positions = data[:, 1:4].astype(np.float32)
        quaternions = data[:, 4:8].astype(np.float32)  # q_w, q_x, q_y, q_z

    elif fmt == 'px4':
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        timestamps = (data[:, 0] * 1000).astype(np.int64)
        positions = data[:, 1:4].astype(np.float32)
        quaternions = np.zeros((len(timestamps), 4), dtype=np.float32)
        quaternions[:, 0] = 1.0  # identity rotation

    return timestamps, positions, quaternions


def quat_conjugate(q):
    """Quaternion conjugate (inverse for unit quaternions). q = [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_rotate_vector(q, v):
    """Rotate vector v by quaternion q. q = [w, x, y, z], v = [x, y, z]."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y + w*z),     2*(x*z - w*y)],
        [    2*(x*y - w*z), 1 - 2*(x*x + z*z),     2*(y*z + w*x)],
        [    2*(x*z + w*y),     2*(y*z - w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)
    return R @ v


def quat_rotate_vectors_batch(q, v):
    """Rotate N vectors by N quaternions (vectorized). q: (N,4) [w,x,y,z], v: (N,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R00 = 1 - 2*(y*y + z*z)
    R01 = 2*(x*y + w*z)
    R02 = 2*(x*z - w*y)
    R10 = 2*(x*y - w*z)
    R11 = 1 - 2*(x*x + z*z)
    R12 = 2*(y*z + w*x)
    R20 = 2*(x*z + w*y)
    R21 = 2*(y*z - w*x)
    R22 = 1 - 2*(x*x + y*y)

    out = np.empty_like(v)
    out[:, 0] = R00 * v[:, 0] + R01 * v[:, 1] + R02 * v[:, 2]
    out[:, 1] = R10 * v[:, 0] + R11 * v[:, 1] + R12 * v[:, 2]
    out[:, 2] = R20 * v[:, 0] + R21 * v[:, 1] + R22 * v[:, 2]

    return out


def align_timestamps(imu_ts, imu_data, gt_ts, gt_positions, gt_quaternions):
    """Align IMU and ground truth by interpolating GT to IMU timestamps.

    Returns:
        imu_data: float32 (M, 6) - trimmed IMU data
        gt_pos_interp: float32 (M, 3) - GT positions at IMU timestamps
        gt_quat_interp: float32 (M, 4) - GT quaternions at IMU timestamps
    """
    t_start = max(imu_ts[0], gt_ts[0])
    t_end = min(imu_ts[-1], gt_ts[-1])

    mask = (imu_ts >= t_start) & (imu_ts <= t_end)
    imu_ts_trim = imu_ts[mask]
    imu_data_trim = imu_data[mask]

    gt_pos_interp = np.zeros((len(imu_ts_trim), 3), dtype=np.float32)
    for ax in range(3):
        gt_pos_interp[:, ax] = np.interp(
            imu_ts_trim.astype(np.float64),
            gt_ts.astype(np.float64),
            gt_positions[:, ax]
        )

    gt_quat_interp = np.zeros((len(imu_ts_trim), 4), dtype=np.float32)
    for ax in range(4):
        gt_quat_interp[:, ax] = np.interp(
            imu_ts_trim.astype(np.float64),
            gt_ts.astype(np.float64),
            gt_quaternions[:, ax]
        )
    norms = np.linalg.norm(gt_quat_interp, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    gt_quat_interp /= norms

    return imu_data_trim, gt_pos_interp, gt_quat_interp


def create_windows(imu_data, gt_positions, gt_quaternions, window_size=200, stride=100):
    """Create sliding windows with body-frame displacement labels.

    The displacement is computed in world frame then rotated into the body
    frame at the start of each window using the ground truth orientation.

    Returns:
        X: float32 (num_windows, window_size, 6) - normalized IMU windows
        y: float32 (num_windows, 3) - body-frame displacement labels (meters)
    """
    imu_norm = imu_data.copy()
    imu_norm[:, :3] /= ACC_NORM
    imu_norm[:, 3:] /= GYRO_NORM

    X_windows = []
    y_displacements = []

    num_windows = (len(imu_norm) - window_size) // stride + 1

    for i in range(num_windows):
        start = i * stride
        end = start + window_size

        X_windows.append(imu_norm[start:end])

        # World-frame displacement
        disp_world = gt_positions[end - 1] - gt_positions[start]

        # Rotate to body frame using orientation at window start
        q_start = gt_quaternions[start]
        q_inv = quat_conjugate(q_start)
        disp_body = quat_rotate_vector(q_inv, disp_world)

        y_displacements.append(disp_body)

    X = np.array(X_windows, dtype=np.float32)
    y = np.array(y_displacements, dtype=np.float32)

    return X, y


def load_tlio_sequence(seq_dir):
    """Load a TLIO golden dataset sequence.

    Reads imu0_resampled.npy (17 columns), rotates IMU from world frame
    to body frame, and returns data compatible with create_windows().

    TLIO .npy columns:
      [0] ts_us, [1-3] gyro_world, [4-6] acc_world,
      [7-10] quat [qx,qy,qz,qw], [11-13] position, [14-16] velocity

    Args:
        seq_dir: Path to sequence directory containing imu0_resampled.npy

    Returns:
        imu_body: float32 (N, 6) - [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z] body frame
        positions: float32 (N, 3) - [x, y, z] world frame (meters)
        quaternions: float32 (N, 4) - [qw, qx, qy, qz]
    """
    npy_path = os.path.join(seq_dir, 'imu0_resampled.npy')
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"TLIO data not found: {npy_path}")

    data = np.load(npy_path)  # (N, 17)
    if data.shape[1] < 17:
        raise ValueError(f"Expected 17 columns in TLIO .npy, got {data.shape[1]}")

    # Extract columns
    gyro_world = data[:, 1:4].astype(np.float32)    # world-frame gyro (rad/s)
    acc_world = data[:, 4:7].astype(np.float32)      # world-frame accel (m/s²)
    quat_xyzw = data[:, 7:11].astype(np.float32)     # [qx, qy, qz, qw]
    positions = data[:, 11:14].astype(np.float32)     # [x, y, z] meters

    # Reorder quaternion: [qx, qy, qz, qw] -> [qw, qx, qy, qz]
    quaternions = np.empty_like(quat_xyzw)
    quaternions[:, 0] = quat_xyzw[:, 3]  # qw
    quaternions[:, 1] = quat_xyzw[:, 0]  # qx
    quaternions[:, 2] = quat_xyzw[:, 1]  # qy
    quaternions[:, 3] = quat_xyzw[:, 2]  # qz

    # Normalize quaternions
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions /= np.maximum(norms, 1e-8)

    # Rotate IMU: world frame -> body frame using conjugate quaternion
    q_conj = quaternions.copy()
    q_conj[:, 1:] *= -1  # conjugate = negate x, y, z

    acc_body = quat_rotate_vectors_batch(q_conj, acc_world)
    gyro_body = quat_rotate_vectors_batch(q_conj, gyro_world)

    # Combine: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    imu_body = np.concatenate([acc_body, gyro_body], axis=1)

    print(f"  TLIO: {len(data)} samples, "
          f"accel mag: {np.mean(np.linalg.norm(acc_body, axis=1)):.2f} m/s², "
          f"gyro mag: {np.mean(np.linalg.norm(gyro_body, axis=1)):.2f} rad/s")

    return imu_body, positions, quaternions


def get_data(data_dir=None, window_size=200, stride=100):
    """Load IMU + ground truth data as numpy arrays.

    Expects data_dir to contain either:
      - imu.csv + groundtruth.csv (single sequence)
      - Subdirectories each with imu.csv + groundtruth.csv (multiple sequences)

    For EuRoC: point to the mav0/ directory containing imu0/data.csv and
    state_groundtruth_estimate0/data.csv

    Args:
        data_dir: Path to data directory (default: ../data/)
        window_size: IMU samples per window
        stride: Step between windows

    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    all_X = []
    all_y = []

    # Check for EuRoC structure
    euroc_imu = os.path.join(data_dir, 'imu0', 'data.csv')
    euroc_gt = os.path.join(data_dir, 'state_groundtruth_estimate0', 'data.csv')

    # Check for simple structure
    simple_imu = os.path.join(data_dir, 'imu.csv')
    simple_gt = os.path.join(data_dir, 'groundtruth.csv')

    if os.path.exists(euroc_imu) and os.path.exists(euroc_gt):
        print(f"Loading EuRoC data from: {data_dir}")
        imu_ts, imu_data = load_imu_csv(euroc_imu)
        gt_ts, gt_pos, gt_quat = load_gt_csv(euroc_gt)
        imu_aligned, gt_pos_aligned, gt_quat_aligned = align_timestamps(
            imu_ts, imu_data, gt_ts, gt_pos, gt_quat)
        X, y = create_windows(imu_aligned, gt_pos_aligned, gt_quat_aligned, window_size, stride)
        all_X.append(X)
        all_y.append(y)
        print(f"  Sequence: {len(imu_aligned)} IMU samples -> {len(X)} windows")

    elif os.path.exists(simple_imu) and os.path.exists(simple_gt):
        print(f"Loading data from: {data_dir}")
        imu_ts, imu_data = load_imu_csv(simple_imu)
        gt_ts, gt_pos, gt_quat = load_gt_csv(simple_gt)
        imu_aligned, gt_pos_aligned, gt_quat_aligned = align_timestamps(
            imu_ts, imu_data, gt_ts, gt_pos, gt_quat)
        X, y = create_windows(imu_aligned, gt_pos_aligned, gt_quat_aligned, window_size, stride)
        all_X.append(X)
        all_y.append(y)
        print(f"  Sequence: {len(imu_aligned)} IMU samples -> {len(X)} windows")

    else:
        subdirs = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])

        if not subdirs:
            raise FileNotFoundError(
                f"No data found in {data_dir}. Expected:\n"
                f"  - {data_dir}/imu.csv + groundtruth.csv, or\n"
                f"  - {data_dir}/imu0/data.csv + state_groundtruth_estimate0/data.csv (EuRoC), or\n"
                f"  - Subdirectories each with imu.csv + groundtruth.csv"
            )

        for subdir in subdirs:
            seq_dir = os.path.join(data_dir, subdir)

            # Try EuRoC structure: subdir/mav0/imu0/data.csv
            imu_path = os.path.join(seq_dir, 'mav0', 'imu0', 'data.csv')
            gt_path = os.path.join(seq_dir, 'mav0', 'state_groundtruth_estimate0', 'data.csv')

            # Try without mav0: subdir/imu0/data.csv
            if not os.path.exists(imu_path):
                imu_path = os.path.join(seq_dir, 'imu0', 'data.csv')
                gt_path = os.path.join(seq_dir, 'state_groundtruth_estimate0', 'data.csv')

            # Try simple: subdir/imu.csv
            if not os.path.exists(imu_path):
                imu_path = os.path.join(seq_dir, 'imu.csv')
                gt_path = os.path.join(seq_dir, 'groundtruth.csv')

            if not os.path.exists(imu_path) or not os.path.exists(gt_path):
                print(f"  Skipping {subdir}: no imu/groundtruth CSVs found")
                continue

            print(f"Loading sequence: {subdir}")
            imu_ts, imu_data = load_imu_csv(imu_path)
            gt_ts, gt_pos, gt_quat = load_gt_csv(gt_path)
            imu_aligned, gt_pos_aligned, gt_quat_aligned = align_timestamps(
                imu_ts, imu_data, gt_ts, gt_pos, gt_quat)
            X, y = create_windows(imu_aligned, gt_pos_aligned, gt_quat_aligned, window_size, stride)
            all_X.append(X)
            all_y.append(y)
            print(f"  {len(imu_aligned)} IMU samples -> {len(X)} windows")

    if not all_X:
        raise FileNotFoundError(f"No valid data sequences found in {data_dir}")

    # Sequence-based split
    num_seqs = len(all_X)
    if num_seqs >= 2:
        split_idx = max(1, int(num_seqs * 0.8))
        X_train = np.concatenate(all_X[:split_idx])
        y_train = np.concatenate(all_y[:split_idx])
        X_test = np.concatenate(all_X[split_idx:])
        y_test = np.concatenate(all_y[split_idx:])
    else:
        X_all = all_X[0]
        y_all = all_y[0]
        split = int(len(X_all) * 0.8)
        X_train, X_test = X_all[:split], X_all[split:]
        y_train, y_test = y_all[:split], y_all[split:]

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    # Quick test
    (X_train, y_train), (X_test, y_test) = get_data()
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Sample IMU (step 0): {X_train[0, 0, :]}")
    print(f"Sample displacement: {y_train[0]} meters")
