import json
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation


def _load_ground_truth(tum_path: Path):
    """Load TUM ground truth file.
    Returns:
        timestamps (np.ndarray, shape (N,)) in seconds
        positions (np.ndarray, shape (N, 3))
        quaternions (np.ndarray, shape (N, 4)) in xyzw order
    """
    timestamps = []
    positions = []
    quaternions = []
    with open(tum_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # expected: ts tx ty tz qx qy qz qw
            if len(parts) != 8:
                raise ValueError(f"Unexpected line format in TUM file: {line}")
            ts = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            timestamps.append(ts)
            positions.append([tx, ty, tz])
            quaternions.append([qx, qy, qz, qw])
    return np.array(timestamps), np.array(positions), np.array(quaternions)


def _load_imu_raw(csv_path: Path):
    """Load raw IMU CSV.
    Returns timestamps (seconds), accel (3,N), gyro (3,N)
    """
    df = pd.read_csv(csv_path, header=None)
    ts = df.iloc[:, 0].to_numpy(dtype=float) * 1e-3
    gyr = df.iloc[:, 2:5].to_numpy().T
    acc = df.iloc[:, 5:8].to_numpy().T
    return ts, acc, gyr


def _apply_time_offset(ts: np.ndarray, offset_sec: float) -> np.ndarray:
    return ts + offset_sec


def _interpolate_measurements(ts_src: np.ndarray, meas_src: np.ndarray, ts_target: np.ndarray) -> np.ndarray:
    """Interpolate 3‑D measurements onto target timestamps.
    meas_src shape (3, N), ts_target shape (M,).
    Returns (3, M).
    """
    interp_fun = interp1d(ts_src, meas_src, axis=1, kind="linear", fill_value="extrapolate")
    return interp_fun(ts_target)


def _compute_velocity(positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Compute velocity using central differences (numpy gradient)."""
    return np.gradient(positions, timestamps, axis=0)


def resample_and_calibrate_imu(
    imu_csv_path: str,
    ground_truth_path: str,
    calibration_json_path: str,
    resample_freq_hz: int,
    not_calibrated: bool = True,
    not_rotated: bool = True,
    not_resampled: bool = True,
    output_npy_path: Optional[str] = None,
) -> np.ndarray:
    """Read raw IMU data, optionally calibrate, rotate, and resample.

    Parameters
    ----------
    imu_csv_path: Path to ``imu_samples_0.csv``.
    ground_truth_path: Path to TUM ``.txt`` file.
    calibration_json_path: Path to ``calibration.json``.
    resample_freq_hz: Target frequency after resampling.
    calibrated, rotated, resampled: Flags controlling each processing step.
    output_npy_path: If provided, the resulting array is saved to this file.

    Returns
    -------
    np.ndarray with columns:
        [ts_us, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z,
         qx, qy, qz, qw, px, py, pz, vx, vy, vz]
    """
    # Load data
    imu_csv_path = Path(imu_csv_path)
    ground_truth_path = Path(ground_truth_path)
    calibration_json_path = Path(calibration_json_path)

    ts_imu, acc_raw, gyr_raw = _load_imu_raw(imu_csv_path)
    gt_ts, gt_pos, gt_quat = _load_ground_truth(ground_truth_path)

    # Calibration
    if not_calibrated:
        with open(calibration_json_path, "r") as f:
            calib_json = json.load(f)
        accel_scale_inv = np.linalg.inv(np.array(calib_json["Accelerometer"]["Model"]["RectificationMatrix"]))
        gyro_scale_inv = np.linalg.inv(np.array(calib_json["Gyroscope"]["Model"]["RectificationMatrix"]))
        accel_bias = np.array(calib_json["Accelerometer"]["Bias"]["Offset"]).reshape((3, 1))
        gyro_bias = np.array(calib_json["Gyroscope"]["Bias"]["Offset"]).reshape((3, 1))
        gyro_gsense = np.zeros((3, 3))
        acc_cal = accel_scale_inv @ acc_raw - accel_bias
        gyr_cal = gyro_scale_inv @ gyr_raw - gyro_gsense @ acc_raw - gyro_bias
    else:
        acc_cal = acc_raw
        gyr_cal = gyr_raw

    # Rotation into world frame
    if not_rotated:
        # Interpolate rotation matrix from GT to IMU timestamps
        # ERROR IT REQUIRES A SINGLE STATIC ROTATION< A SINGLE UNIFIED FRAME, THIS PRODUCES INCONSISTENT ROTATIONS TOO ADAPTED
        rot_mats = Rotation.from_quat(gt_quat).as_matrix()  # (N,3,3)
        rot_interp = np.empty((len(ts_imu), 3, 3))
        for i in range(3):
            for j in range(3):
                rot_interp[:, i, j] = interp1d(gt_ts, rot_mats[:, i, j], kind="linear", fill_value="extrapolate")(ts_imu)
        acc_world = np.einsum("...ij,...j->...i", rot_interp, acc_cal.T).T
        gyr_world = np.einsum("...ij,...j->...i", rot_interp, gyr_cal.T).T
    else:
        acc_world = acc_cal
        gyr_world = gyr_cal

    # Apply any timestamp offset from the calibration file (fallback 0)
    calib_dict = calib_json if not_calibrated else {}
    offset = float(calib_dict.get("TimeOffsetSec_Device_Accel", 0.0))
    ts_imu_aligned = _apply_time_offset(ts_imu, offset)
    #I PTODUCES EXACT FREQUENCY ADAPTED DATASET, MAY BE A RIGID VERSION OF THE ORIGINAL IMPLEMENTATION 
    if not_resampled:
        start, end = gt_ts[0], gt_ts[-1]
        dt = 1.0 / float(resample_freq_hz)
        uniform_ts = np.arange(start, end, dt)
        acc_interp = _interpolate_measurements(ts_imu_aligned, acc_world, uniform_ts)
        gyr_interp = _interpolate_measurements(ts_imu_aligned, gyr_world, uniform_ts)
        pos_interp = interp1d(gt_ts, gt_pos, axis=0, kind="linear", fill_value="extrapolate")(uniform_ts)
        quat_interp = interp1d(gt_ts, gt_quat, axis=0, kind="linear", fill_value="extrapolate")(uniform_ts)
        vel_interp = _compute_velocity(pos_interp, uniform_ts)
        ts_us = (uniform_ts * 1e6).astype(np.int64)
    else:
        ts_us = (ts_imu_aligned * 1e6).astype(np.int64)
        acc_interp = acc_world
        gyr_interp = gyr_world
        pos_interp = interp1d(gt_ts, gt_pos, axis=0, kind="linear", fill_value="extrapolate")(ts_imu_aligned)
        quat_interp = interp1d(gt_ts, gt_quat, axis=0, kind="linear", fill_value="extrapolate")(ts_imu_aligned)
        vel_interp = _compute_velocity(pos_interp, ts_imu_aligned)

    N = ts_us.shape[0]
    result = np.empty((N, 17), dtype=np.float64)
    result[:, 0] = ts_us
    result[:, 1:4] = acc_interp.T
    result[:, 4:7] = gyr_interp.T
    result[:, 7:11] = quat_interp
    result[:, 11:14] = pos_interp
    result[:, 14:17] = vel_interp

    if output_npy_path:
        np.save(output_npy_path, result)
    return result
