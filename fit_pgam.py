from core import compute_spike_rates, compute_sniff_freqs_bins, align_brain_and_behavior, load_behavior
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import time
import sys
sys.path.append('src/PGAM')

import GAM_library as gl
import gam_data_handlers as gdh
from post_processing import postprocess_results
import yaml
import statsmodels.api as sm




def preprocess(data_dir, save_dir, mouse, session, window_size, step_size, use_units, nfs = 30_000, sfs = 1_000):

    # Loading the neural data and computing the spike rates
    kilosort_dir = os.path.join(data_dir, 'kilosorted', mouse, session)
    rates_OB, rates_HC, time_bins, ob_units, hc_units = compute_spike_rates(kilosort_dir, nfs, window_size, step_size, use_units=use_units, sigma = 0, zscore=False)
    rates = np.concatenate((rates_HC, rates_OB), axis=0)
    units = np.concatenate((hc_units, ob_units), axis=0)


    # Loading the sniffing data
    sniff_params_file = os.path.join(data_dir, 'sniff events', mouse, session, 'sniff_params')
    mean_freqs, latencies, phases = compute_sniff_freqs_bins(sniff_params_file, time_bins, window_size, sfs)


    # Loading the behavior (tracking & task variable) data
    behavior_dir = os.path.join(data_dir, 'behavior_data', mouse, session)
    tracking_dir = os.path.join(data_dir, 'sleap_predictions', mouse, session)
    tracking_file = os.path.join(tracking_dir, next(f for f in os.listdir(tracking_dir) if f.endswith('.analysis.h5')))
    events = load_behavior(behavior_dir, tracking_file)


    # Aligning the neural and behavior data
    rates_data = align_brain_and_behavior(events, rates, units, time_bins, window_size)
    rates_data = rates_data.assign(sns=mean_freqs, latency=latencies, phase=phases)
    rates_data['sns'] = rates_data['sns'].interpolate(method='linear')
    rates_data.dropna(subset=['x', 'y', 'v_x', 'v_y'], inplace=True)
    print(rates_data.head())


    # Converting the data to numpy arrays for PGAM standardized input
    counts = np.array(rates_data.drop(columns=['x', 'y', 'v_x', 'v_y', 'sns', 'speed', 'latency', 'phase', 'reward_state', 'time', 'trial_id', 'click']).values) * window_size
    position = rates_data[['x', 'y']].to_numpy()
    velocity = rates_data[['v_x', 'v_y']].to_numpy()
    rates_data.drop(columns=['x', 'y', 'v_x', 'v_y'], inplace=True)
    variables = [
        position,                                 # shape (N, 2)
        velocity,                                 # shape (N, 2)
        rates_data['sns'].to_numpy(),            # shape (N,)
        rates_data['latency'].to_numpy(),
        rates_data['phase'].to_numpy(),
        rates_data['speed'].to_numpy(),
        rates_data['click'].to_numpy()
    ]

    variable_names = ['position', 'velocity', 'sns', 'latency', 'phase', 'speed', 'click']

    trial_ids = np.array(rates_data['trial_id'].values)

    neu_names = np.array(rates_data.columns[:len(counts[0])])

    neu_info = {}
    for i, name in enumerate(neu_names):
        neu_info[name] = {'area': 'HC' if i < len(hc_units) else 'OB', 'id': units[i]}


    # Plot the variables
    plot_dir = os.path.join(save_dir, 'behaior_figs')
    os.makedirs(plot_dir, exist_ok=True)
    for i, name in enumerate(variable_names):
        if name in ['position', 'velocity']:
            plt.figure(figsize=(15, 8))
            plt.plot(variables[i][:, 0], label=f'{name} x')
            plt.plot(variables[i][:, 1], label=f'{name} y')
            plt.title(name)
            plt.legend()
            sns.despine()
            plt.savefig(os.path.join(plot_dir, f'{name}.png'))
        else:
            plt.figure(figsize=(15, 8))
            plt.plot(variables[i])
            plt.title(name)
            sns.despine()
            plt.savefig(os.path.join(plot_dir, f'{name}.png'))

    for i, v in enumerate(variables):
        print(f"Variable {variable_names[i]}: shape = {v.shape}", flush=True)


    #np.savez(os.path.join(save_dir, f'data.npz'), counts=counts, variables=variables, variable_names=variable_names, trial_ids = trial_ids, neu_names = neu_names, neu_info=neu_info)
    return counts, variables, variable_names, trial_ids, neu_names, neu_info


def make_config(order, window_size, save_path):

    # make the knots
    knots_x = np.hstack(([-75]*(order-1), np.linspace(-75,75,5),[75]*(order-1)))
    knots_y = np.hstack(([-150]*(order-1), np.linspace(-150,150,5),[150]*(order-1)))
    knots_v_x = np.hstack(([-10]*(order-1), np.linspace(-10,10,5),[10]*(order-1)))
    knots_v_y = np.hstack(([-10]*(order-1), np.linspace(-10,10,5),[10]*(order-1)))
    knots_sns = np.hstack(([2]*(order-1), np.linspace(2,12,5),[12]*(order-1)))
    knots_latency = np.hstack(([0]*(order-1), np.linspace(0,1,5),[1.25]*(order-1)))
    knots_phase = np.linspace(0, 2*np.pi, 5)
    knots_speed = np.hstack(([0]*(order-1), np.linspace(0,10,5),[10]*(order-1)))

    # convert to float
    knots_x = [float(k) for k in knots_x]
    knots_y = [float(k) for k in knots_y]
    knots_v_x = [float(k) for k in knots_v_x]
    knots_v_y = [float(k) for k in knots_v_y]
    knots_sns = [float(k) for k in knots_sns]
    knots_latency = [float(k) for k in knots_latency]
    knots_phase = [float(k) for k in knots_phase]
    knots_speed = [float(k) for k in knots_speed]

    # create the config dictionary
    cov_dict = {
        'position' : {
            'lam':10, 
            'penalty_type': 'der', 
            'der': 2, 
            'knots': [knots_x, knots_y],
            'order':order,
            'is_temporal_kernel': False,
            'is_cyclic': [False, False],
            'knots_num': np.nan,
            'kernel_length': np.nan,
            'kernel_direction': np.nan,
            'samp_period':window_size
        },
        'velocity' : {
            'lam':10,
            'penalty_type': 'der',
            'der': 2,
            'knots': [knots_v_x, knots_v_y],
            'order':order,
            'is_temporal_kernel': False,
            'is_cyclic': [False, False],
            'knots_num': np.nan,
            'kernel_length': np.nan,
            'kernel_direction': np.nan,
            'samp_period':window_size
        },
        'sns' : {
            'lam':10, 
            'penalty_type': 'der', 
            'der': 2, 
            'knots': knots_sns,
            'order':order,
            'is_temporal_kernel': False,
            'is_cyclic': [False],
            'knots_num': np.nan,
            'kernel_length': np.nan,
            'kernel_direction': np.nan,
            'samp_period':window_size 
        },
        'latency' : {
            'lam':10, 
            'penalty_type': 'der', 
            'der': 2, 
            'knots': knots_latency,
            'order':order,
            'is_temporal_kernel': False,
            'is_cyclic': [False],
            'knots_num': np.nan,
            'kernel_length': np.nan,
            'kernel_direction': np.nan,
            'samp_period':window_size 
        },
        'phase' : {
            'lam':10, 
            'penalty_type': 'der', 
            'der': 2, 
            'knots': knots_phase,
            'order':order,
            'is_temporal_kernel': False,
            'is_cyclic': [True],
            'knots_num': np.nan,
            'kernel_length': np.nan,
            'kernel_direction': np.nan,
            'samp_period':window_size 
        },
        'speed': {
            'lam':10, 
            'penalty_type': 'der', 
            'der': 2, 
            'knots': knots_speed,
            'order':order,
            'is_temporal_kernel': False,
            'is_cyclic': [False],
            'knots_num': np.nan,
            'kernel_length': np.nan,
            'kernel_direction': np.nan,
            'samp_period':window_size
        },
        'click' : {
            'lam':10, 
            'penalty_type': 'der', 
            'der': 2, 
            'knots': np.nan,
            'order':order,
            'is_temporal_kernel': True,
            'is_cyclic': [False],
            'knots_num': 8,
            'kernel_length': 101,
            'kernel_direction': 0,
            'samp_period':window_size
        }
    }

    # save the yaml config
    with open(os.path.join(save_path, 'config.yml'), 'w') as outfile:
        yaml.dump(cov_dict, outfile, default_flow_style=False)



def main(args):


    # Use a non-interactive backend for matplotlib
    matplotlib.use('Agg')


    # Unpacking the arguments which were initially passed from the bash script submit_PGAM.sh
    data_dir = args.data_dir
    save_dir = args.save_dir
    mouse = args.mouse
    session = args.session
    window_size = args.window_size
    step_size = args.window_step
    use_units = args.use_units
    order = args.order
    frac_eval = args.frac_eval


    # Updating the save directory for the current mouse and session
    save_dir = os.path.join(save_dir, mouse, session)

    # loading the config file 
    with open(os.path.join(save_dir, 'config.yml'), 'r') as stream:
        config_dict = yaml.safe_load(stream)

    # Preprocessing the data
    start = time.time()
    counts, variables, variable_names, trial_ids, neu_names, neu_info = preprocess(data_dir, save_dir, mouse, session, window_size, step_size, use_units)
    print(f'Preprocessing complete in {time.time()-start:.2f} seconds')

    # Fit the PGAM on all the neurons
    for fit_num, neuron_num in enumerate(neu_names):
        start = time.time()
        
        # create the train and eval sets. Eval are not
        train_trials = trial_ids % (np.round(1/frac_eval)) != 0
        eval_trials = ~train_trials

        # create and populate the smooth handler object
        sm_handler = gdh.smooths_handler()
        for var in config_dict.keys():

            # check if var is a neuron or a variable
            if var in variable_names:
                idx = variable_names.index(var)
                x_var = variables[idx]
            elif var in neu_names:
                x_var = np.squeeze(counts[:, np.array(neu_names) == var])
            else:
                raise ValueError('Variable "%s" not found in the input data!'%var)
            
            raw_knots = config_dict[var]['knots']
            if raw_knots is None or (isinstance(raw_knots, float) and np.isnan(raw_knots)):
                knots = None
            elif isinstance(raw_knots[0], (list, np.ndarray)):
                knots = [np.array(k) for k in raw_knots]  # tensor product
            else:
                knots = [np.array(raw_knots)]  # single 1D variable

            lam = config_dict[var]['lam']
            penalty_type = config_dict[var]['penalty_type']
            der = config_dict[var]['der']
            order = config_dict[var]['order']  
            is_temporal_kernel = config_dict[var]['is_temporal_kernel']
            is_cyclic =  config_dict[var]['is_cyclic']
            knots_num = config_dict[var]['knots_num']
            kernel_length = config_dict[var]['kernel_length']
            kernel_direction = config_dict[var]['kernel_direction']
            samp_period = config_dict[var]['samp_period']
            
            # rename the variable as spike hist if the input is the spike counts of the neuron we are fitting
            if var == neuron_num:
                label = 'spike_hist'
            else:
                label = var
                

            if x_var.ndim == 2 and x_var.shape[1] == 2:
                x_var = [x_var[:, 0], x_var[:, 1]]  # list of two 1D arrays
            else:
                x_var = [x_var]  # still wrap 1D variables in a list

            # Add the variable to the smooth handler
            sm_handler.add_smooth(label, x_var, knots=knots, ord=order, is_temporal_kernel=is_temporal_kernel,
                            trial_idx=trial_ids, is_cyclic=is_cyclic, penalty_type=penalty_type, der=der, lam=lam,
                                knots_num=knots_num, kernel_length=kernel_length,kernel_direction=kernel_direction,
                                time_bin=samp_period)
        
        # Set the link function (log link) and the distribution family (Poisson) for the spike counts
        link = sm.genmod.families.links.log()
        poissFam = sm.genmod.families.Poisson(link=link)
        spk_counts = np.squeeze(counts[:, fit_num])

        # Create the PGAM model
        pgam = gl.general_additive_model(sm_handler, sm_handler.smooths_var, spk_counts, poissFam)
        print(f'\n\nFitting neuron {neuron_num} ({fit_num+1}/{len(neu_names)})', flush=True)
        print(f'Neuron information: {neu_info[neuron_num]}', flush=True)
        full, reduced = pgam.fit_full_and_reduced(sm_handler.smooths_var, 
                                        th_pval=1e-5,# pval for significance of covariate inclusion
                                        max_iter=1e2,
                                        use_dgcv=True,
                                        trial_num_vec=trial_ids,
                                        filter_trials=train_trials,)
        print('\nMinimal subset of variables driving the activity:', flush=True)
        if reduced is None:
            print('No significant variables found')
            var_list = []
        else:
            var_list = reduced.var_list
            print(var_list, flush=True)

        # saving the list of variables driving the activity
        np.savez(os.path.join(save_dir, f'{neuron_num}_var_list.npz'), var_list=var_list, unit_info=neu_info[neuron_num])
        



        # This only works when fitting 1D variables!
        post_process = False
        if post_process:
            print('\npost-process fit results...')
            res = postprocess_results(neuron_num, spk_counts, full, reduced, train_trials,
                            sm_handler, poissFam, trial_ids, var_zscore_par=None, info_save=neu_info, bins=100)
            
            np.savez(os.path.join(save_dir, f'{neuron_num}_results.npz'), results = res)

            # plot tuning functions
            plt.figure(figsize=(18,8))
            plt.title('Tuning functions for neuron %s'%neuron_num)
            nk = len(res['variable'])
            for k in range(nk):
                plt.subplot(2,nk,k+1)
                plt.title('log-space %s'%res['variable'][k])
                x_kernel = res['x_kernel'][k]
                y_kernel = res['y_kernel'][k]
                ypCI_kernel = res['y_kernel_pCI'][k]
                ymCI_kernel = res['y_kernel_mCI'][k]
                plt.plot(x_kernel, y_kernel, color='r')
                plt.fill_between(x_kernel, ymCI_kernel, ypCI_kernel, color='r', alpha=0.3)
                x_firing = res['x_rate_Hz'][k]
                y_firing_model = res['model_rate_Hz'][k]
                y_firing_raw = res['raw_rate_Hz'][k]
                plt.subplot(2,nk,k+nk+1)
                plt.title('rate-space %s'%res['variable'][k])
                plt.plot(x_firing, y_firing_raw, color='grey',label='raw')
                plt.plot(x_firing, y_firing_model, color='r',label='model')
                if k == 0:
                    plt.legend()
                plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{neuron_num}_tuning.png'))
            plt.close()
            print('Neuron %s done in %.2f seconds'%(neuron_num, time.time()-start))
            print('\n\n')









if __name__ == "__main__":
    master_clock = time.time()
    
    # Unpacking the parameters from the bash script submit_PGAM.sh
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='PGAM_data')
    parser.add_argument('--save_dir', type=str, default='PGAM_results')
    parser.add_argument('--mouse', type=str, default='6000')
    parser.add_argument('--session', type=str, default='1')
    parser.add_argument('--window_size', type=float, default=0.01,
                        help='Window size for spike counts in seconds')
    parser.add_argument('--window_step', type=float, default=0.01,
                        help='Window step for spike counts in seconds')
    parser.add_argument('--use_units', type=str, choices=['good', 'good/mua', 'mua'], default='good/mua',
                        help='What kilosort cluster labels to use')
    parser.add_argument('--order', type=int, default=4,
                        help='Order of the B-splines to fit to the data')
    parser.add_argument('--frac_eval', type=float, default=0.2,
                        help='Fraction of trials to use for evaluation')
    args = parser.parse_args()

    # Make the config file for the model
    config_save_path = os.path.join(args.save_dir, args.mouse, args.session)
    os.makedirs(config_save_path, exist_ok=True)
    make_config(args.order, args.window_size, config_save_path)

    # Fit the model
    main(args)
    print(f'Analysis complete in {time.time()-master_clock:.2f} seconds')
