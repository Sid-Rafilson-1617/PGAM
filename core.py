import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt
import seaborn as sns





def compute_spike_rates(kilosort_dir: str, sampling_rate: int, window_size: float = 1.0, step_size: float = 0.5, use_units: str = 'all', sigma: float = 2.5, zscore: bool = True):
    
    """
    Compute smoothed spike rates for neural units in OB (olfactory bulb) and HC (hippocampus) regions 
    using a sliding window approach from Kilosort output data.
    
    This function processes spike times and cluster assignments from Kilosort/Phy2, separates units by 
    brain region based on channel mapping, calculates firing rates within sliding time windows, and 
    applies Gaussian smoothing. Optionally, z-scoring can be applied to normalize firing rates.
    
    Parameters
    ----------
    kilosort_dir : str
        Path to the directory containing Kilosort output files.
    sampling_rate : int
        Sampling rate of the recording in Hz.
    window_size : float, optional
        Size of the sliding window in seconds, default is 1.0.
    step_size : float, optional
        Step size for sliding window advancement in seconds, default is 0.5.
    use_units : str, optional
        Filter for unit types to include:
        - 'all': Include all units
        - 'good': Include only good units
        - 'mua': Include only multi-unit activity
        - 'good/mua': Include both good units and multi-unit activity
        - 'noise': Include only noise units
        Default is 'all'.
    sigma : float, optional
        Standard deviation for Gaussian smoothing kernel, default is 2.5.
    zscore : bool, optional
        Whether to z-score the firing rates, default is True.
    
    Returns
    -------
    spike_rate_matrix_OB : ndarray
        Matrix of spike rates for OB units (shape: num_OB_units × num_windows).
    spike_rate_matrix_HC : ndarray
        Matrix of spike rates for HC units (shape: num_HC_units × num_windows).
    time_bins : ndarray
        Array of starting times for each window.
    ob_units : ndarray
        Array of unit IDs for OB region.
    hc_units : ndarray
        Array of unit IDs for HC region.
    
    Notes
    -----
    - OB units are assumed to be on channels 16-31
    - HC units are assumed to be on channels 0-15
    - Firing rates are computed in Hz (spikes per second)
    
    Raises
    ------
    FileNotFoundError
        If any required Kilosort output files are missing.
    """    

    # Load spike times and cluster assignments
    spike_times_path = os.path.join(kilosort_dir, "spike_times.npy")
    spike_clusters_path = os.path.join(kilosort_dir, "spike_clusters.npy")  # Cluster assignments from Phy2 manual curation
    templates_path = os.path.join(kilosort_dir, "templates.npy")
    templates_ind_path = os.path.join(kilosort_dir, "templates_ind.npy")
    cluster_groups_path = os.path.join(kilosort_dir, "cluster_group.tsv")

    # Ensure all required files exist
    if not all(os.path.exists(p) for p in [spike_times_path, spike_clusters_path, templates_path, templates_ind_path, cluster_groups_path]):
        raise FileNotFoundError("Missing required Kilosort output files.")

    # Loading the data
    templates = np.load(templates_path)  # Shape: (nTemplates, nTimePoints, nChannels)
    templates_ind = np.load(templates_ind_path)  # Shape: (nTemplates, nChannels)
    spike_times = np.load(spike_times_path) / sampling_rate  # Convert to seconds
    spike_clusters = np.load(spike_clusters_path)
    cluster_groups = np.loadtxt(cluster_groups_path, dtype=str, skiprows=1, usecols=[1])

    # Find peak amplitude channel for each template and assign to unit
    peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    unit_best_channels = {unit: templates_ind[unit, peak_channels[unit]] for unit in range(len(peak_channels))}
    
    # Filter units based on use_units parameter
    if use_units == 'all':
        unit_best_channels = unit_best_channels
    elif use_units == 'good':
        unit_indices = np.where(cluster_groups == 'good')[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}
    elif use_units == 'mua':
        unit_indices = np.where(cluster_groups == 'mua')[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}
    elif use_units == 'good/mua':
        unit_indices = np.where(np.isin(cluster_groups, ['good', 'mua']))[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}
    elif use_units == 'noise':
        unit_indices = np.where(cluster_groups == 'noise')[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}
    else:
        raise ValueError(f"Unknown unit selection '{use_units}'.")


    # Get total duration of the recording
    recording_duration = np.max(spike_times)

    # Define time windows
    time_bins = np.arange(0, recording_duration - window_size, step_size)
    num_windows = len(time_bins)

    # Separate OB and HC units
    hc_units = np.array([unit for unit, ch in unit_best_channels.items() if ch in range(0, 16)])
    ob_units = np.array([unit for unit, ch in unit_best_channels.items() if ch in range(16, 32)])
    num_ob_units = len(ob_units)
    num_hc_units = len(hc_units)

    # Initialize spike rate matrices
    spike_rate_matrix_OB = np.zeros((num_ob_units, num_windows))
    spike_rate_matrix_HC = np.zeros((num_hc_units, num_windows))

    # Compute spike counts in each window
    for i, t_start in enumerate(time_bins):
        t_end = t_start + window_size

        # Find spikes in this window
        in_window = (spike_times >= t_start) & (spike_times < t_end)
        spike_clusters_in_window = spike_clusters[in_window]

        # Compute spike rates for OB
        for j, unit in enumerate(ob_units):
            spike_rate_matrix_OB[j, i] = np.sum(spike_clusters_in_window == unit) / window_size  # Hz

        # Compute spike rates for HC
        for j, unit in enumerate(hc_units):
            spike_rate_matrix_HC[j, i] = np.sum(spike_clusters_in_window == unit) / window_size  # Hz

    # Apply Gaussian smoothing
    for j in range(num_ob_units):
        if sigma > 0:
            spike_rate_matrix_OB[j, :] = gaussian_filter1d(spike_rate_matrix_OB[j, :], sigma=sigma)

    for j in range(num_hc_units):
        if sigma > 0:
            spike_rate_matrix_HC[j, :] = gaussian_filter1d(spike_rate_matrix_HC[j, :], sigma=sigma)

    # Apply Z-scoring (optional)
    if zscore:
        def z_score(matrix):
            mean_firing = np.mean(matrix, axis=1, keepdims=True)
            std_firing = np.std(matrix, axis=1, keepdims=True)
            std_firing[std_firing == 0] = 1  # Prevent division by zero
            return (matrix - mean_firing) / std_firing

        spike_rate_matrix_OB = z_score(spike_rate_matrix_OB)
        spike_rate_matrix_HC = z_score(spike_rate_matrix_HC)

    return spike_rate_matrix_OB, spike_rate_matrix_HC, time_bins, ob_units, hc_units



def compute_sniff_freqs_bins(
    sniff_params_file: str,
    time_bins: np.ndarray,
    window_size: float,
    sfs: int,
):
    """Compute sniffing statistics aligned to provided time bins.

    Parameters
    ----------
    sniff_params_file : str
        Path to the MATLAB ``sniff_params`` file.
    time_bins : np.ndarray
        Start times of the neural data bins in seconds.
    window_size : float
        Width of each time bin in seconds.
    sfs : int
        Sampling frequency of the sniff signal (Hz).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Mean sniff frequency, latency from the last inhalation and sniff phase for
        each time bin.
    """

    inhalation_times, _, exhalation_times, _ = load_sniff_MATLAB(sniff_params_file)
    inhalation_times = inhalation_times / sfs  # Convert to seconds

    # Compute sniff frequencies
    freqs = 1 / np.diff(inhalation_times)  # Instantaneous frequency between inhalations

    # Remove unrealistic frequencies
    bad_indices = np.where((freqs > 14) | (freqs < 0.8))[0]
    freqs = np.delete(freqs, bad_indices)
    inhalation_times = np.delete(inhalation_times[:-1], bad_indices)  # Align with freqs

    # Initialize outputs
    mean_freqs = np.full(len(time_bins), np.nan)
    inhalation_latencies = np.full(len(time_bins), np.nan)
    phases = np.full(len(time_bins), np.nan)

    for i, t_start in enumerate(time_bins):
        t_end = t_start + window_size
        middle = t_start + window_size / 2

        in_window = (inhalation_times >= t_start) & (inhalation_times < t_end)

        # Find last inhalation before the center of the bin
        last_idx = np.where(inhalation_times < middle)[0]
        if len(last_idx) > 0:
            last_idx = last_idx[-1]
            last_inh_time = inhalation_times[last_idx]
            inhalation_latencies[i] = middle - last_inh_time

            # Phase: time since last inhalation / duration of current sniff
            if last_idx < len(freqs):  # Ensure freq is defined
                sniff_duration = 1 / freqs[last_idx]
                phase_fraction = (middle - last_inh_time) / sniff_duration
                phases[i] = (phase_fraction % 1) * 2 * np.pi  # Convert to radians
        else:
            inhalation_latencies[i] = np.nan
            phases[i] = np.nan

        # Mean frequency in the bin
        if np.any(in_window):
            mean_freqs[i] = np.nanmean(freqs[in_window])

    return mean_freqs, inhalation_latencies, phases



def align_brain_and_behavior(events: pd.DataFrame, spike_rates: np.ndarray, units: np.ndarray, time_bins: np.ndarray, window_size: float = 0.1, speed_threshold: float = 100, interp_method = 'linear', order = None):
    
    """
    Align neural spike rate data with behavioral tracking data using time windows.
    
    This function matches neural activity from spike rates with behavioral metrics (position, velocity, speed)
    by finding the closest behavioral event to the middle of each time bin. It creates a unified dataframe
    containing both neural and behavioral data, removes outliers based on speed threshold, and interpolates
    missing values.
    
    Parameters
    ----------
    events : pd.DataFrame
        Behavioral tracking data containing columns:
        - 'timestamp_ms': Timestamps in milliseconds
        - 'centroid_x', 'centroid_y': Position coordinates
        - 'velocity_x', 'velocity_y': Velocity components
        - 'speed': Overall movement speed
    
    spike_rates : np.ndarray
        Matrix of spike rates with shape (n_units, n_time_bins).
    
    units : np.ndarray
        Array of unit IDs corresponding to rows in spike_rates.
    
    time_bins : np.ndarray
        Array of starting times for each time bin in seconds.
    
    window_size : float, optional
        Size of each time window in seconds, default is 0.1.
    
    speed_threshold : float, optional
        Maximum allowed speed in cm/s for behavioral measurements. Values above this
        threshold are treated as outliers. Default is 100.
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe with aligned neural and behavioral data containing:
        - Unit columns: Spike rates for each neural unit
        - 'x', 'y': Position coordinates
        - 'v_x', 'v_y': Velocity components
        - 'speed': Movement speed
        - 'time': Time bin start times
        
    Notes
    -----
    - For each time bin, the behavioral event closest to the middle of the bin is selected
    - Speed outliers are removed using an absolute threshold
    - Missing values are interpolated using linear interpolation
    - Rows with missing behavioral data (typically at beginning/end of recording) are removed
    """

    # Initialize arrays for holding aligned data
    mean_positions_x = np.full(len(time_bins), np.nan)
    mean_positions_y = np.full(len(time_bins), np.nan)
    mean_velocities_x = np.full(len(time_bins), np.nan)
    mean_velocities_y = np.full(len(time_bins), np.nan)
    mean_speeds = np.full(len(time_bins), np.nan)
    mean_rewards = np.full(len(time_bins), np.nan)

    # getting event times in seconds
    event_times = events['timestamp_ms'].values / 1000

    # Calculate mean behavior in each time bin
    for i, t_start in enumerate(time_bins):
        t_end = t_start + window_size
        middle = t_start + window_size / 2

        if np.any(event_times < middle):
            nearest_event_index = np.argmin(np.abs(event_times - middle))
            mean_positions_x[i] = events['position_x'].iloc[nearest_event_index]
            mean_positions_y[i] = events['position_y'].iloc[nearest_event_index]
            mean_velocities_x[i] = events['velocity_x'].iloc[nearest_event_index]
            mean_velocities_y[i] = events['velocity_y'].iloc[nearest_event_index]
            mean_speeds[i] = events['speed'].iloc[nearest_event_index]
            mean_rewards[i] = events['reward_state'].iloc[nearest_event_index]
        else:
            mean_positions_x[i] = np.nan
            mean_positions_y[i] = np.nan
            mean_velocities_x[i] = np.nan
            mean_velocities_y[i] = np.nan
            mean_speeds[i] = np.nan
            mean_rewards[i] = np.nan


    # converting the spike rate matrix to a DataFrame
    data = pd.DataFrame(spike_rates.T, columns=[f"Unit {i}" for i in units])

    # adding the tracking data to the DataFrame
    conversion = 5.1
    data['x'] = mean_positions_x / conversion # convert to cm
    data['y'] = mean_positions_y / conversion # convert to cm
    data['v_x'] = mean_velocities_x / conversion # convert to cm/s
    data['v_y'] = mean_velocities_y / conversion # convert to cm/s
    data['speed'] = mean_speeds / conversion # convert to cm/s
    data['time'] = time_bins # in seconds
    data['reward_state'] = mean_rewards

    # Remove samples where speed exceeds the allowable threshold
    data.loc[data['speed'] > speed_threshold, ['x', 'y', 'v_x', 'v_y', 'speed']] = np.nan

    # interpolating the tracking data to fill in NaN values
    data.interpolate(method=interp_method, inplace=True, order = order)

    # Finding the trial number and getting the click time
    trial_ids = np.zeros(data.shape[0])
    click_event = np.zeros(data.shape[0])
    for i in range(1, len(data)):
        trial_ids[i] = trial_ids[i-1]
        if data['reward_state'].iloc[i-1] and not data['reward_state'].iloc[i]:
            trial_ids[i] += 1
            click_event[i] = 1
    data = data.assign(trial_id = trial_ids, click = click_event)

    return data



def load_behavior(behavior_file: str, tracking_file: str = None) -> pd.DataFrame:

    """
    Load and preprocess behavioral tracking data from a CSV file.
    
    This function loads movement tracking data, normalizes spatial coordinates by
    centering them around zero, calculates velocity components and overall speed,
    and returns a filtered dataframe with relevant movement metrics.
    
    Parameters
    ----------
    behavior_file : str
        Path to the directory containing ``events.csv`` with behavioral tracking
        data. The file should include columns for ``centroid_x``, ``centroid_y``
        and ``timestamp_ms``.
        
    Returns
    -------
    events : pandas.DataFrame
        Processed dataframe containing the following columns:
        - ``position_x`` and ``position_y``: Zero-centered coordinates
        - ``velocity_x`` and ``velocity_y``: Rate of change in position
        - ``reward_state``: Binary reward indicator
        - ``speed``: Overall movement speed (Euclidean norm of velocity components)
        - ``timestamp_ms``: Timestamps in milliseconds
        
    Notes
    -----
    - Position coordinates are normalized by subtracting the mean to center around zero
    - Velocity is calculated using first-order differences (current - previous position)
    - The first velocity value uses the first position value as the "previous" position
    - Speed is calculated as the Euclidean distance between consecutive positions
    """

    # Load the behavior data
    events = pd.read_csv(os.path.join(behavior_file, 'events.csv'))

    if tracking_file:
        # Load the SLEAP tracking data from the HDF5 file
        f = h5py.File(tracking_file, 'r')
        nose = f['tracks'][:].T[:, 0, :]
        nose = nose[:np.shape(events)[0], :]
        mean_x, mean_y = np.nanmean(nose[:, 0]), np.nanmean(nose[:, 1])
        events['position_x'] = nose[:, 0] - mean_x
        events['position_y'] = nose[:, 1] - mean_y
        
    else:
        # zero-mean normalize the x and y coordinates
        mean_x, mean_y = np.nanmean(events['centroid_x']), np.nanmean(events['centroid_y'])
        events['position_x'] = events['centroid_x'] - mean_x
        events['position_y'] = events['centroid_y'] - mean_y

    # Estimating velocity and speed
    events['velocity_x'] = np.diff(events['position_x'], prepend=events['position_x'].iloc[0])
    events['velocity_y'] = np.diff(events['position_y'], prepend=events['position_y'].iloc[0])
    events['speed'] = np.sqrt(events['velocity_x']**2 + events['velocity_y']**2)



    # keeping only the columns we need
    events = events[['position_x', 'position_y', 'velocity_x', 'velocity_y', 'reward_state', 'speed', 'timestamp_ms']]
    return events



def load_sniff_MATLAB(file: str) -> np.array:
    '''
    Loads a MATLAB file containing sniff data and returns a numpy array
    '''

    mat = loadmat(file)
    sniff_params = mat['sniff_params']

    # loading sniff parameters
    inhalation_times = sniff_params[:, 0]
    inhalation_voltage = sniff_params[:, 1]
    exhalation_times = sniff_params[:, 2]
    exhalation_voltage = sniff_params[:, 3]

    # bad sniffs are indicated by 0 value in exhalation_times
    bad_indices = np.where(exhalation_times == 0)


    # removing bad sniffs
    inhalation_times = np.delete(inhalation_times, bad_indices)
    inhalation_voltage = np.delete(inhalation_voltage, bad_indices)
    exhalation_times = np.delete(exhalation_times, bad_indices)
    exhalation_voltage = np.delete(exhalation_voltage, bad_indices)

    return inhalation_times.astype(np.int32), inhalation_voltage, exhalation_times.astype(np.int32), exhalation_voltage


def preprocess(
    data_dir,
    save_dir,
    mouse,
    session,
    window_size,
    step_size,
    use_units,
    nfs=30_000,
    sfs=1_000,
):
    """Preprocess neural, sniff and behavioral data for PGAM fitting."""

    # --- Neural data: compute spike rates ---
    kilosort_dir = os.path.join(data_dir, "kilosorted", mouse, session)
    rates_OB, rates_HC, time_bins, ob_units, hc_units = compute_spike_rates(
        kilosort_dir, nfs, window_size, step_size, use_units=use_units, sigma=0, zscore=False
    )
    rates = np.concatenate((rates_HC, rates_OB), axis=0)
    units = np.concatenate((hc_units, ob_units), axis=0)

    # --- Sniffing data ---
    sniff_params_file = os.path.join(data_dir, "sniff", mouse, session, "sniff_params")
    mean_freqs, latencies, phases = compute_sniff_freqs_bins(
        sniff_params_file, time_bins, window_size, sfs
    )

    # --- Behavioral data ---
    behavior_dir = os.path.join(data_dir, "behavior_data", mouse, session)
    tracking_dir = os.path.join(data_dir, "sleap_predictions", mouse, session)
    tracking_file = os.path.join(
        tracking_dir, next(f for f in os.listdir(tracking_dir) if f.endswith(".analysis.h5"))
    )
    events = load_behavior(behavior_dir, tracking_file)

    # --- Align neural and behavioral data ---
    rates_data = align_brain_and_behavior(events, rates, units, time_bins, window_size)
    rates_data = rates_data.assign(sns=mean_freqs, latency=latencies, phase=phases)
    rates_data["sns"] = rates_data["sns"].interpolate(method="linear")
    rates_data.dropna(subset=["x", "y", "v_x", "v_y"], inplace=True)
    print(rates_data.head())

    # --- Convert to standardized PGAM input ---
    counts = (
        np.array(
            rates_data.drop(
                columns=[
                    "x",
                    "y",
                    "v_x",
                    "v_y",
                    "sns",
                    "speed",
                    "latency",
                    "phase",
                    "reward_state",
                    "time",
                    "trial_id",
                    "click",
                ]
            ).values
        )
        * window_size
    )
    variables = [
        rates_data["x"].to_numpy(),
        rates_data["y"].to_numpy(),
        rates_data["v_x"].to_numpy(),
        rates_data["v_y"].to_numpy(),
        rates_data["sns"].to_numpy(),
        rates_data["latency"].to_numpy(),
        rates_data["phase"].to_numpy(),
        rates_data["speed"].to_numpy(),
        rates_data["click"].to_numpy(),
    ]
    rates_data.drop(columns=["x", "y", "v_x", "v_y"], inplace=True)

    variable_names = [
        "position_x",
        "position_y",
        "velocity_x",
        "velocity_y",
        "sns",
        "latency",
        "phase",
        "speed",
        "click",
    ]

    trial_ids = np.array(rates_data["trial_id"].values)
    neu_names = np.array(rates_data.columns[: len(counts[0])])

    neu_info = {}
    for i, name in enumerate(neu_names):
        neu_info[name] = {"area": "HC" if i < len(hc_units) else "OB", "id": units[i]}

    # --- Plot variables for inspection ---
    plot_dir = os.path.join(save_dir, "behavior_figs")
    os.makedirs(plot_dir, exist_ok=True)
    for i, name in enumerate(variable_names):
        if name in ["position", "velocity"]:
            plt.figure(figsize=(15, 8))
            plt.plot(variables[i][:, 0], label=f"{name} x")
            plt.plot(variables[i][:, 1], label=f"{name} y")
            plt.title(name)
            plt.legend()
            sns.despine()
            plt.savefig(os.path.join(plot_dir, f"{name}.png"))
            plt.close()
        else:
            plt.figure(figsize=(15, 8))
            plt.plot(variables[i])
            plt.title(name)
            sns.despine()
            plt.savefig(os.path.join(plot_dir, f"{name}.png"))
            plt.close()

    # np.savez(
    #     os.path.join(save_dir, f"data.npz"),
    #     counts=counts,
    #     variables=variables,
    #     variable_names=variable_names,
    #     trial_ids=trial_ids,
    #     neu_names=neu_names,
    #     neu_info=neu_info,
    # )
    return counts, variables, variable_names, trial_ids, neu_names, neu_info

