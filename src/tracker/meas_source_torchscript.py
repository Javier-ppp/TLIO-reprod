import numpy as np
import torch
from network.covariance_parametrization import DiagonalParam
from utils.logging import logging


class MeasSourceTorchScript:
    """ Loading a torchscript has the advantage that we do not need to reconstruct the original network class to
        load the weights, the network structure is contained into the torchscript file.
    """
    def __init__(self, model_path, force_cpu=False):
        # load trained network model
        logging.info("Loding {}...".format(model_path))
        
    
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.net = torch.jit.load(model_path, map_location="mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.net = torch.jit.load(model_path, map_location="cuda:0")
        else:
            torch.init_num_threads()
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            self.device = torch.device("cpu")
            self.net = torch.jit.load(model_path, map_location="cpu")

            
                
            # NOTE TLIO baseline model won't work on GPU unless we ass map_location
            # https://github.com/pytorch/pytorch/issues/78207
            self.net = torch.jit.load(model_path, map_location=self.device)

        self.net.to(self.device)
        self.net.eval()
        logging.info("Model {} loaded to device {}.".format(model_path, self.device))

    def get_displacement_measurement(self, net_gyr_w, net_acc_w, clip_small_disp=False):
        with torch.no_grad():
            features = np.concatenate([net_gyr_w, net_acc_w], axis=1)  # N x 6
            features_t = torch.unsqueeze(
                torch.from_numpy(features.T).float().to(self.device), 0
            )  # 1 x 6 x N
            
            netargs = [features_t]
            outputs = self.net(*netargs)

            if type(outputs) == tuple:  # Legacy
                meas, meas_cov = outputs
            elif type(outputs) == dict:  # New output format
                meas, meas_cov = outputs["pred"], outputs["pred_log_std"]
            else:
                # Handle other output formats
                logging.warning(f"Unexpected output type: {type(outputs)}")
                if isinstance(outputs, torch.Tensor):
                    logging.warning(f"Tensor shape: {outputs.shape}, dim: {outputs.dim()}")
                    # Assume it's just the measurement
                    meas = outputs
                    meas_cov = torch.zeros_like(meas)
                else:
                    raise ValueError(f"Unsupported output format from model: {type(outputs)}")

            # Handle 3D measurement tensors (sequence of measurements over window)
            # If this is the case, the network predicts over the whole window at high frequency.
            # TODO utilize the whole window measurements. May improve.
            if meas.dim() == 3:
                meas = meas[:, :, -1]  # Take last time step: [B, 3, T] -> [B, 3]
                if meas_cov.dim() == 3:
                    meas_cov = meas_cov[:, :, -1]

            assert meas.dim() == 2, f"Expected meas to be 2D, got shape {meas.shape}, dim {meas.dim()}"  # [B,3]
            assert meas_cov.dim() == 2

            meas = meas.cpu().detach().numpy()
            meas_cov[meas_cov < -4] = -4  # exp(-3) =~ 0.05
            meas_cov = DiagonalParam.vec2Cov(meas_cov).cpu().detach().numpy()[0, :, :]
            meas = meas.reshape((3, 1))

            # Our equivalent of zero position update (TODO need stronger prior to keep it still)
            if clip_small_disp and np.linalg.norm(meas) < 0.001:
                meas = 0 * meas
                # meas_cov = 1e-6 * np.eye(3)

            return meas, meas_cov
