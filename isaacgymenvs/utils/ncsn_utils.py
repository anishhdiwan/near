import argparse
from collections import deque
import torch
import copy
import numpy as np

def to_relative_pose(pose_traj, root_idx):
    """Transform a vector of body poses to be relative to the root joint and return the flattened vectors

    Args:
        pose_traj (torch.Tensor): Tensor of a body pose trajectory. Required shape: [num_frames, num_joints, 3]
        root_idx (int): index of the root joint. Must be in [0,num_joints)
    """
    assert len(pose_traj.shape) == 3, "Required trajectory shape: [num_frames, num_joints, 3]"
    assert 0 <= root_idx < pose_traj.shape[1], "Root joint index must be in [0, num_joints)"

    # Subtract the root body cartesian pose from the other joints
    return (pose_traj - pose_traj[:, root_idx, :].unsqueeze(1)).flatten(start_dim=1, end_dim=-1)

def get_series_derivative(series, dt):
    """Compute the derivative of a time series diven a time step length dt

    Args:
        series (torch.Tensor): time series of shape [d, dim]
        dt (float): time step
    """
    if series.shape[0] < 2:
        return torch.zeros((1, series.shape[-1]))

    else:
        series_shift = copy.deepcopy(series)
        series_shift[:-1,:] = series_shift.clone()[1:,:]
        series_shift = series_shift[:-1]
        series = series[:-1]
        derivative = (series_shift - series)/dt

        return derivative

def dict2namespace(config):
    """Convert a disctionary (typically containing config params) to a namespace structure (https://tedboy.github.io/python_stdlib/generated/generated/argparse.Namespace.html#argparse.Namespace)

    Args:
        config (dict): dictionary of configs params
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class LastKMovingAvg:
    """Given a time series where data is continously being added, keep a buffer of the last k elements and return their stats

    Assumes that the elements are torch tensors
    """

    def __init__(self, maxlen=3):
        self.maxlen = maxlen
        self.buffer = deque([], maxlen=self.maxlen)
    
    def append(self, element, return_avg=True):
        self.buffer.append(element)

        if return_avg:
            return torch.mean(torch.stack(list(self.buffer))).item()

    def reset(self):
        self.buffer = deque([], maxlen=self.maxlen)



def sparc(movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Borrowed from https://github.com/siva82kb/SPARC/blob/3650934f5bc9ad21c7d59dcdb6ccd249c462b3e4/scripts/smoothness.py#L8
    
    Calcualtes the smoothness of the given speed profile using the modified
    spectral arc length metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]

    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magntiude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.

    Notes
    -----
    This is the modfieid spectral arc length metric, which has been tested only
    for discrete movements.

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> sal, _, _ = sparc(move, fs=100.)
    >>> '%.5f' % sal
    '-1.41403'

    """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Indices to choose only the spectrum within the given cut off frequency
    # Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)