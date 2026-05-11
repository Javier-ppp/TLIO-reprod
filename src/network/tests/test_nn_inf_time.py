import json
import os
from os import path as osp
import time

import torch
from dataloader.memmapped_sequences_dataset import MemMappedSequencesDataset
from ..model_factory import get_model
from torch.utils.data import DataLoader

from utils.dotdict import dotdict
from utils.utils import to_device
from utils.logging import logging
from utils.math_utils import *


def benchmark_sequence_generation(
    network,
    data_loader,
    device,
):
    """
    Benchmark ONLY inference timing.

    Returns:
        total_sequence_time
        avg_token_time
        total_tokens
    """

    network.eval()

    total_time = 0.0
    total_tokens = 0

    with torch.no_grad():

        for sample in data_loader:

            sample = to_device(sample, device)

            feat = sample["feats"]["imu0"]

            # -----------------------------------
            # Synchronize before timing
            # -----------------------------------

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            start_time = time.perf_counter()

            pred, pred_cov = network(feat)

            # -----------------------------------
            # Synchronize after timing
            # -----------------------------------

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            elapsed = time.perf_counter() - start_time

            total_time += elapsed

            # -----------------------------------
            # Token counting
            # -----------------------------------

            # Seq2seq output:
            # [B, C, T]
            if len(pred.shape) == 3:
                batch_tokens = pred.shape[0] * pred.shape[2]

            # Standard output:
            # [B, C]
            elif len(pred.shape) == 2:
                batch_tokens = pred.shape[0]

            else:
                raise ValueError(
                    f"Unexpected prediction shape: {pred.shape}"
                )

            total_tokens += batch_tokens

    avg_token_time = total_time / total_tokens

    return {
        "total_sequence_generation_time_s": float(total_time),
        "average_token_generation_time_s": float(avg_token_time),
        "total_tokens_generated": int(total_tokens),
    }
def arg_conversion(args):
    """ Conversions from time arguments to data size """

    if not (args.past_time * args.imu_freq).is_integer():
        raise ValueError(
            "past_time cannot be represented by integer number of IMU data."
        )
    if not (args.window_time * args.imu_freq).is_integer():
        raise ValueError(
            "window_time cannot be represented by integer number of IMU data."
        )
    if not (args.future_time * args.imu_freq).is_integer():
        raise ValueError(
            "future_time cannot be represented by integer number of IMU data."
        )
    if not (args.imu_freq / args.sample_freq).is_integer():
        raise ValueError("sample_freq must be divisible by imu_freq.")

    data_window_config = dotdict()
    data_window_config.past_data_size = int(args.past_time * args.imu_freq)
    data_window_config.window_size = int(args.window_time * args.imu_freq)
    data_window_config.future_data_size = int(args.future_time * args.imu_freq)
    data_window_config.step_size = int(args.imu_freq / args.sample_freq)
    data_window_config.data_style = "resampled"
    data_window_config.input_sensors = ["imu0"]
    data_window_config.decimator = 10
    data_window_config.express_in_t0_yaw_normalized_frame = False

    net_config = {
        "in_dim": (
            data_window_config["past_data_size"]
            + data_window_config["window_size"]
            + data_window_config["future_data_size"]
        )
        // 32
        + 1
    }

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info(f"Training/testing with {args.imu_freq} Hz IMU data")
    logging.info(
        "Size: "
        + str(data_window_config["past_data_size"])
        + "+"
        + str(data_window_config["window_size"])
        + "+"
        + str(data_window_config["future_data_size"])
        + ", "
        + "Time: "
        + str(args.past_time)
        + "+"
        + str(args.window_time)
        + "+"
        + str(args.future_time)
    )
    logging.info("Perturb on bias: %s" % args.do_bias_shift)
    logging.info("Perturb on gravity: %s" % args.perturb_gravity)
    logging.info("Sample frequency: %s" % args.sample_freq)
    return data_window_config, net_config

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list

def net_test(args):
    """
    Benchmark ONLY inference timing.
    No plots.
    No trajectory metrics.
    No loss metrics.
    """

    try:

        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")

        if args.out_dir is not None:

            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)

            logging.info(f"Benchmark output writes to {args.out_dir}")

        else:
            raise ValueError("out_dir must be specified.")

        data_window_config, net_config = arg_conversion(args)

    except ValueError as e:
        logging.error(e)
        return

    test_list_path = osp.join(args.root_dir, "test_list.txt")
    test_list = get_datalist(test_list_path)

    # ---------------------------------------------------------
    # Device selection
    # ---------------------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    checkpoint = torch.load(
        args.model_path,
        map_location=device,
    )

    network = get_model(
        args.arch,
        net_config,
        args.input_dim,
        args.output_dim,
    ).to(device)

    network.load_state_dict(
        checkpoint["model_state_dict"]
    )

    network.eval()

    logging.info(
        f"Model {args.model_path} loaded to device {device}."
    )

    # ---------------------------------------------------------
    # Benchmark results container
    # ---------------------------------------------------------

    all_metrics = {}

    # ---------------------------------------------------------
    # Iterate sequences
    # ---------------------------------------------------------

    for data in test_list:

        logging.info(f"Benchmarking {data}...")

        try:

            seq_dataset = MemMappedSequencesDataset(
                args.root_dir,
                "test",
                data_window_config,
                sequence_subset=[data],
                store_in_ram=True,
            )

            seq_loader = DataLoader(
                seq_dataset,
                batch_size=1024,
                shuffle=False,
            )

        except OSError as e:
            print(e)
            continue

        # -----------------------------------------------------
        # Benchmark timing only
        # -----------------------------------------------------

        metrics = benchmark_sequence_generation(
            network,
            seq_loader,
            device,
        )

        logging.info(metrics)

        all_metrics[data] = metrics

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------

    try:

        outfile = osp.join(
            args.out_dir,
            "benchmark_metrics.json",
        )

        with open(outfile, "w") as f:
            json.dump(all_metrics, f, indent=1)

        logging.info(f"Saved benchmark to {outfile}")

    except Exception as e:
        logging.error(e)

    return

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="test")

    parser.add_argument("--root_dir", type=str, required=True)

    parser.add_argument("--arch", type=str, required=True)

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--imu_freq", type=float, default=200.0)

    parser.add_argument("--past_time", type=float, default=0.0)

    parser.add_argument("--window_time", type=float, default=1.0)

    parser.add_argument("--future_time", type=float, default=0.0)

    parser.add_argument("--sample_freq", type=float, default=20.0)

    parser.add_argument("--input_dim", type=int, default=6)

    parser.add_argument("--output_dim", type=int, default=3)

    parser.add_argument("--rpe_window", type=float, default=2.0)

    parser.add_argument("--do_bias_shift", action="store_true")

    parser.add_argument("--perturb_gravity", action="store_true")

    args = parser.parse_args()

    net_test(args)