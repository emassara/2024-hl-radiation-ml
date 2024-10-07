import argparse
import sys
import pprint
import time
import datetime
import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates
from tqdm import tqdm
import shutil
import traceback
import matplotlib.animation as animation
import glob
import sunpy.visualization.colormaps as sunpycm

from datasets import SDOMLlite, SDOCore, RadLab, GOESXRS, GOESSGPS, Sequences, UnionDataset
from models import RadRecurrent, RadRecurrentWithSDO, RadRecurrentWithSDOCore
from events_months import EventCatalog

matplotlib.use('Agg')

sdo_cms = {}
sdo_cms['hmi_m'] = sunpycm.cmlist.get('hmimag')
sdo_cms['aia_0131'] = sunpycm.cmlist.get('sdoaia131')
sdo_cms['aia_0171'] = sunpycm.cmlist.get('sdoaia171')
sdo_cms['aia_0193'] = sunpycm.cmlist.get('sdoaia193')
sdo_cms['aia_0211'] = sunpycm.cmlist.get('sdoaia211')
sdo_cms['aia_1600'] = sunpycm.cmlist.get('sdoaia1600')


class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def save_model(model, optimizer, epoch, iteration, train_losses, valid_losses, file_name):
    print('Saving model to {}'.format(file_name))
    if isinstance(model, RadRecurrent):
        checkpoint = {
            'model': 'RadRecurrent',
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'model_context_window': model.context_window,
            'model_prediction_window': model.prediction_window,
            'model_data_dim': model.data_dim,
            'model_lstm_dim': model.lstm_dim,
            'model_lstm_depth': model.lstm_depth,
            'model_dropout': model.dropout
        }
    elif isinstance(model, RadRecurrentWithSDO):
        checkpoint = {
            'model': 'RadRecurrentWithSDO',
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'model_context_window': model.context_window,
            'model_prediction_window': model.prediction_window,
            'model_data_dim_context': model.data_dim_context,
            'model_data_dim_prediction': model.data_dim_predict,
            'model_lstm_dim': model.lstm_dim,
            'model_lstm_depth': model.lstm_depth,
            'model_dropout': model.dropout,
            'model_sdo_channels': model.sdo_channels,
            'model_sdo_dim': model.sdo_dim
        }
    else:
        raise ValueError('Unknown model type: {}'.format(model))
    torch.save(checkpoint, file_name)


def load_model(file_name, device):
    checkpoint = torch.load(file_name, weights_only=False)
    if checkpoint['model'] == 'RadRecurrent':    
        model_context_window = checkpoint['model_context_window']
        model_prediction_window = checkpoint['model_prediction_window']
        model_data_dim = checkpoint['model_data_dim']
        model_lstm_dim = checkpoint['model_lstm_dim']
        model_lstm_depth = checkpoint['model_lstm_depth']
        model_dropout = checkpoint['model_dropout']
        model = RadRecurrent(data_dim=model_data_dim, lstm_dim=model_lstm_dim, lstm_depth=model_lstm_depth, dropout=model_dropout, context_window=model_context_window, prediction_window=model_prediction_window)
    elif checkpoint['model'] == 'RadRecurrentWithSDO':
        model_context_window = checkpoint['model_context_window']
        model_prediction_window = checkpoint['model_prediction_window']
        model_data_dim_context = checkpoint['model_data_dim_context']
        model_data_dim_prediction = checkpoint['model_data_dim_prediction']
        model_lstm_dim = checkpoint['model_lstm_dim']
        model_lstm_depth = checkpoint['model_lstm_depth']
        model_dropout = checkpoint['model_dropout']
        model_sdo_channels = checkpoint['model_sdo_channels']
        model_sdo_dim = checkpoint['model_sdo_dim']
        model = RadRecurrentWithSDO(data_dim_context=model_data_dim_context, 
                                    data_dim_predict=model_data_dim_prediction,
                                    lstm_dim=model_lstm_dim, lstm_depth=model_lstm_depth, dropout=model_dropout, context_window=model_context_window, prediction_window=model_prediction_window, sdo_channels=model_sdo_channels, sdo_dim=model_sdo_dim)
    else:
        raise ValueError('Unknown model type: {}'.format(checkpoint['model']))

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']

    return model, optimizer, epoch, iteration, train_losses, valid_losses


def seed(seed=None):
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    print('Setting seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_test_file(prediction_dates, biosentinel_predictions, goesxrs_ground_truth_dates, goesxrs_ground_truth_values, biosentinel_ground_truth_dates, biosentinel_ground_truth_values, file_name):
    print('Saving test results: {}'.format(file_name))

    # goessgps10_prediction_mean = np.mean(goessgps10_predictions, axis=0)
    # goessgps10_prediction_std = np.std(goessgps10_predictions, axis=0)

    # goessgps100_prediction_mean = np.mean(goessgps100_predictions, axis=0)
    # goessgps100_prediction_std = np.std(goessgps100_predictions, axis=0)

    # goesxrs_prediction_mean = np.mean(goesxrs_predictions, axis=0)
    # goesxrs_prediction_std = np.std(goesxrs_predictions, axis=0)

    biosentinel_prediction_mean = np.mean(biosentinel_predictions, axis=0)
    biosentinel_prediction_std = np.std(biosentinel_predictions, axis=0)

    with open(file_name, 'w') as f:
        f.write('date,biosentinel_prediction_mean,biosentinel_prediction_std,goesxrs_ground_truth,biosentinel_ground_truth\n')
        for i in range(len(prediction_dates)):
            date = prediction_dates[i]
            # goessgps10_prediction_mean_value = goessgps10_prediction_mean[i]
            # goessgps10_prediction_std_value = goessgps10_prediction_std[i]
            # goessgps100_prediction_mean_value = goessgps100_prediction_mean[i]
            # goessgps100_prediction_std_value = goessgps100_prediction_std[i]
            # goesxrs_prediction_mean_value = goesxrs_prediction_mean[i]
            # goesxrs_prediction_std_value = goesxrs_prediction_std[i]
            biosentinel_prediction_mean_value = biosentinel_prediction_mean[i]
            biosentinel_prediction_std_value = biosentinel_prediction_std[i]

            # if date in goessgps10_ground_truth_dates:
            #     goessgps10_ground_truth_value = goessgps10_ground_truth_values[goessgps10_ground_truth_dates.index(date)]
            # else:
            #     goessgps10_ground_truth_value = float('nan')

            # if date in goessgps100_ground_truth_dates:
            #     goessgps100_ground_truth_value = goessgps100_ground_truth_values[goessgps100_ground_truth_dates.index(date)]
            # else:
            #     goessgps100_ground_truth_value = float('nan')

            if date in goesxrs_ground_truth_dates:
                goesxrs_ground_truth_value = goesxrs_ground_truth_values[goesxrs_ground_truth_dates.index(date)]
            else:
                goesxrs_ground_truth_value = float('nan')

            if date in biosentinel_ground_truth_dates:
                biosentinel_ground_truth_value = biosentinel_ground_truth_values[biosentinel_ground_truth_dates.index(date)]
            else:
                biosentinel_ground_truth_value = float('nan')

            f.write('{},{},{},{},{}\n'.format(date, #goessgps10_prediction_mean_value, goessgps10_prediction_std_value, goessgps100_prediction_mean_value, goessgps100_prediction_std_value, goesxrs_prediction_mean_value, goesxrs_prediction_std_value, 
                                            biosentinel_prediction_mean_value, biosentinel_prediction_std_value, #goessgps10_ground_truth_value, goessgps100_ground_truth_value, 
                                            goesxrs_ground_truth_value, biosentinel_ground_truth_value))
            

def save_test_plot(context_dates, prediction_dates, training_prediction_window_end, biosentinel_predictions, goesxrs_ground_truth_dates, goesxrs_ground_truth_values, biosentinel_ground_truth_dates, biosentinel_ground_truth_values, file_name, title=None):
    print('Saving test plot: {}'.format(file_name))
    fig, axs = plt.subplot_mosaic([['biosentinel'],['goesxrs']], figsize=(20, 10), height_ratios=[1,1])

    num_samples = goesxrs_predictions.shape[0]

    ylims = {}
    hours_locator = matplotlib.dates.HourLocator(interval=1)
    colors = {}
    colors['biosentinel'] = 'mediumblue'
    # colors['goessgps10'] = 'darkgreen'
    # colors['goessgps100'] = 'darkred'
    colors['goesxrs'] = 'purple'
    colors['prediction'] = 'red'
    prediction_alpha = 0.08
    prediction_mean_alpha = 0.66

    ax = axs['biosentinel']
    ax.set_title('Biosentinel BPD')
    ax.set_ylabel('Absorbed dose rate\n[μGy/min]')
    ax.yaxis.set_label_position("right")
    ax.plot(biosentinel_ground_truth_dates, biosentinel_ground_truth_values, color=colors['biosentinel'], label='Ground truth', alpha=0.75)
    ax.plot(prediction_dates, np.mean(biosentinel_predictions, axis=0), color=colors['prediction'], alpha=prediction_mean_alpha, label='Prediction (mean)')
    for i in range(num_samples):
        label = 'Prediction (samples)' if i == 0 else None
        ax.plot(prediction_dates, biosentinel_predictions[i], label=label, color=colors['prediction'], alpha=prediction_alpha)
    ax.xaxis.set_minor_locator(hours_locator)
    ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
    ax.grid(color='lightgray', zorder=0, which='major')
    ax.set_xticklabels([])
    ax.set_yscale('log')    
    ax.axvline(context_dates[0], color=colors['prediction'], linestyle='--', linewidth=1)
    ax.axvline(prediction_dates[0], color=colors['prediction'], linestyle='-', linewidth=1.5)
    ax.axvline(training_prediction_window_end, color=colors['prediction'], linestyle='--', linewidth=1)
    ax.legend(loc='upper right')
    ylims['biosentinel'] = ax.get_ylim()

    # ax = axs['goessgps10']
    # ax.set_title('GOES SGPS >10 MeV')
    # ax.set_ylabel('Proton flux\n[part./(cm^2 s sr)]')
    # ax.yaxis.set_label_position("right")
    # ax.plot(goessgps10_ground_truth_dates, goessgps10_ground_truth_values, color=colors['goessgps10'], label='Ground truth', alpha=0.75)
    # ax.plot(prediction_dates, np.mean(goessgps10_predictions, axis=0), color=colors['prediction'], alpha=prediction_mean_alpha, label='Prediction (mean)')
    # for i in range(num_samples):
    #     label = 'Prediction (samples)' if i == 0 else None
    #     ax.plot(prediction_dates, goessgps10_predictions[i], label=label, color=colors['prediction'], alpha=prediction_alpha)
    # ax.xaxis.set_minor_locator(hours_locator)
    # ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
    # ax.grid(color='lightgray', zorder=0, which='major')
    # ax.set_xticklabels([])
    # ax.set_yscale('log')
    # ax.set_xticks(axs['biosentinel'].get_xticks())
    # ax.set_xlim(axs['biosentinel'].get_xlim())
    # major_formatter = matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')
    # ax.xaxis.set_major_formatter(major_formatter)
    # ax.axvline(context_dates[0], color=colors['prediction'], linestyle='--', linewidth=1)
    # ax.axvline(prediction_dates[0], color=colors['prediction'], linestyle='-', linewidth=1.5)
    # ax.axvline(training_prediction_window_end, color=colors['prediction'], linestyle='--', linewidth=1)
    # ax.legend(loc='upper right')
    # ylims['goessgps10'] = ax.get_ylim()

    # ax = axs['goessgps100']
    # ax.set_title('GOES SGPS >100 MeV')
    # ax.set_ylabel('Proton flux\n[part./(cm^2 s sr)]')
    # ax.yaxis.set_label_position("right")
    # ax.plot(goessgps100_ground_truth_dates, goessgps100_ground_truth_values, color=colors['goessgps100'], label='Ground truth', alpha=0.75)
    # ax.plot(prediction_dates, np.mean(goessgps100_predictions, axis=0), color=colors['prediction'], alpha=prediction_mean_alpha, label='Prediction (mean)')
    # for i in range(num_samples):
    #     label = 'Prediction (samples)' if i == 0 else None
    #     ax.plot(prediction_dates, goessgps100_predictions[i], label=label, color=colors['prediction'], alpha=prediction_alpha)
    # ax.xaxis.set_minor_locator(hours_locator)
    # ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
    # ax.grid(color='lightgray', zorder=0, which='major')
    # ax.set_xticklabels([])
    # ax.set_yscale('log')
    # ax.set_xticks(axs['biosentinel'].get_xticks())
    # ax.set_xlim(axs['biosentinel'].get_xlim())
    # major_formatter = matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')
    # ax.xaxis.set_major_formatter(major_formatter)
    # ax.axvline(context_dates[0], color=colors['prediction'], linestyle='--', linewidth=1)
    # ax.axvline(prediction_dates[0], color=colors['prediction'], linestyle='-', linewidth=1.5)
    # ax.axvline(training_prediction_window_end, color=colors['prediction'], linestyle='--', linewidth=1)
    # ax.legend(loc='upper right')
    # ylims['goessgps100'] = ax.get_ylim()

    ax = axs['goesxrs']
    ax.set_title('GOES XRS')
    ax.set_ylabel('X-ray flux\n[W/m^2]')
    ax.yaxis.set_label_position("right")
    ax.plot(goesxrs_ground_truth_dates, goesxrs_ground_truth_values, color=colors['goesxrs'], label='Ground truth', alpha=0.75)
    # ax.plot(prediction_dates, np.mean(goesxrs_predictions, axis=0), color=colors['prediction'], alpha=prediction_mean_alpha, label='Prediction (mean)')
    # for i in range(num_samples):
    #     label = 'Prediction (samples)' if i == 0 else None
    #     ax.plot(prediction_dates, goesxrs_predictions[i], label=label, color=colors['prediction'], alpha=prediction_alpha)
    ax.xaxis.set_minor_locator(hours_locator)
    ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
    ax.grid(color='lightgray', zorder=0, which='major')
    # ax.set_xticklabels([])
    ax.set_yscale('log')
    ax.set_xticks(axs['biosentinel'].get_xticks())
    ax.set_xlim(axs['biosentinel'].get_xlim())
    major_formatter = matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')
    ax.xaxis.set_major_formatter(major_formatter)
    ax.axvline(context_dates[0], color=colors['prediction'], linestyle='--', linewidth=1)
    ax.axvline(prediction_dates[0], color=colors['prediction'], linestyle='-', linewidth=1.5)
    ax.axvline(training_prediction_window_end, color=colors['prediction'], linestyle='--', linewidth=1)
    ax.legend(loc='upper right')
    ylims['goesxrs'] = ax.get_ylim()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if title is not None:
        plt.suptitle(title + ' ' + str(prediction_dates[0]))
    plt.savefig(file_name)

    return ylims


def run_test(model, date_start, date_end, file_prefix, title, args):
    data_dir_sdo = os.path.join(args.data_dir, args.sdo_dir)
    # data_dir_goes_sgps = os.path.join(args.data_dir, args.goes_sgps_file)
    data_dir_goes_xrs = os.path.join(args.data_dir, args.goes_xrs_file)
    data_dir_radlab = os.path.join(args.data_dir, args.radlab_file)

    # predict start

    context_start = date_start - datetime.timedelta(minutes=(model.context_window - 1) * args.delta_minutes)
    # dataset_goes_sgps10 = GOESSGPS(data_dir_goes_sgps, date_start=context_start, date_end=date_end, column='>10MeV')
    # dataset_goes_sgps100 = GOESSGPS(data_dir_goes_sgps, date_start=context_start, date_end=date_end, column='>100MeV')
    dataset_goes_xrs = GOESXRS(data_dir_goes_xrs, date_start=context_start, date_end=date_end, random_data=args.xray_random_data)
    dataset_rad = RadLab(data_dir_radlab, instrument=args.rad_inst, date_start=context_start, date_end=date_end, random_data=args.rad_random_data)
    if isinstance(model, RadRecurrentWithSDO):
        dataset_sdo = SDOMLlite(data_dir_sdo, main_study_dir, date_start=context_start, date_end=date_end, random_data=args.sdo_random_data)
        dataset_sequences = Sequences([dataset_sdo, dataset_goes_xrs, dataset_rad], delta_minutes=args.delta_minutes, sequence_length=model.context_window)
        if len(dataset_sequences) == 0:
            return
        context_sequence = dataset_sequences[0]
        context_dates = [datetime.datetime.fromisoformat(d) for d in context_sequence[3]]
    elif isinstance(model, RadRecurrent):
        dataset_sequences = Sequences([dataset_goes_sgps10, dataset_goes_sgps100, dataset_goes_xrs, dataset_rad], delta_minutes=args.delta_minutes, sequence_length=model.context_window)
        if len(dataset_sequences) == 0:
            return
        context_sequence = dataset_sequences[0]
        context_dates = [datetime.datetime.fromisoformat(d) for d in context_sequence[4]]
    else:
        raise ValueError('Unknown model type: {}'.format(model))

    if context_start != context_dates[0]:
        print('context start adjusted from {} to {} (due to data availability)'.format(context_start, context_dates[0]))
    context_start = context_dates[0]
    context_end = context_dates[-1]

    prediction_window = int((date_end - context_end).total_seconds() / (args.delta_minutes * 60))

    if isinstance(model, RadRecurrent):
        context_goessgps10 = context_sequence[0][:model.context_window].unsqueeze(1).to(args.device)
        context_goessgps100 = context_sequence[1][:model.context_window].unsqueeze(1).to(args.device)
        context_goesxrs = context_sequence[2][:model.context_window].unsqueeze(1).to(args.device)
        context_rad = context_sequence[3][:model.context_window].unsqueeze(1).to(args.device)
        context = torch.cat([context_goessgps10, context_goessgps100, context_goesxrs, context_rad], dim=1)
        context_batch = context.unsqueeze(0).repeat(args.num_samples, 1, 1)
        prediction_batch = model.predict(context_batch, prediction_window).detach()
    elif isinstance(model, RadRecurrentWithSDO):
        context_sdo = context_sequence[0][:model.context_window].to(args.device)
        context_sdo_batch = context_sdo.unsqueeze(0)
        # context_goessgps10 = context_sequence[1][:model.context_window].unsqueeze(1).to(args.device)
        # context_goessgps100 = context_sequence[2][:model.context_window].unsqueeze(1).to(args.device)
        context_goesxrs = context_sequence[1][:model.context_window].unsqueeze(1).to(args.device)
        context_rad = context_sequence[2][:model.context_window].unsqueeze(1).to(args.device)
        context_data = torch.cat([context_goesxrs, context_rad], dim=1)
        context_data_batch = context_data.unsqueeze(0)
        prediction_batch = model.predict(context_sdo_batch, context_data_batch, prediction_window, num_samples=args.num_samples).detach()
    else:
        raise ValueError('Unknown model type: {}'.format(model))

    prediction_date_start = context_end
    prediction_dates = [prediction_date_start + datetime.timedelta(minutes=i*args.delta_minutes) for i in range(prediction_window + 1)]
    training_prediction_window_end = prediction_date_start + datetime.timedelta(minutes=model.prediction_window*args.delta_minutes)

    # goessgps10_predictions = prediction_batch[:, :, 0]
    # goessgps100_predictions = prediction_batch[:, :, 1]
    # goesxrs_predictions = prediction_batch[:, :, 2]
    biosentinel_predictions = prediction_batch[:, :, 0]

    # goessgps10_predictions = dataset_goes_sgps10.unnormalize_data(goessgps10_predictions).cpu().numpy()
    # goessgps100_predictions = dataset_goes_sgps100.unnormalize_data(goessgps100_predictions).cpu().numpy()
    # goesxrs_predictions = dataset_goes_xrs.unnormalize_data(goesxrs_predictions).cpu().numpy()
    biosentinel_predictions = dataset_rad.unnormalize_data(biosentinel_predictions).cpu().numpy()

    # predict end

    # goessgps10_ground_truth_dates, goessgps10_ground_truth_values = dataset_goes_sgps10.get_series(context_start, date_end, delta_minutes=args.delta_minutes)
    # goessgps100_ground_truth_dates, goessgps100_ground_truth_values = dataset_goes_sgps100.get_series(context_start, date_end, delta_minutes=args.delta_minutes)
    goesxrs_ground_truth_dates, goesxrs_ground_truth_values = dataset_goes_xrs.get_series(context_start, date_end, delta_minutes=args.delta_minutes)
    biosentinel_ground_truth_dates, biosentinel_ground_truth_values = dataset_rad.get_series(context_start, date_end, delta_minutes=args.delta_minutes)

    # goessgps10_ground_truth_values = dataset_goes_sgps10.unnormalize_data(goessgps10_ground_truth_values)
    # goessgps100_ground_truth_values = dataset_goes_sgps100.unnormalize_data(goessgps100_ground_truth_values)
    goesxrs_ground_truth_values = dataset_goes_xrs.unnormalize_data(goesxrs_ground_truth_values)
    biosentinel_ground_truth_values = dataset_rad.unnormalize_data(biosentinel_ground_truth_values)

    file_name = os.path.join(args.target_dir, file_prefix)
    test_file = file_name + '.csv'
    save_test_file(prediction_dates, #goessgps10_predictions, goessgps100_predictions, goesxrs_predictions, 
                    biosentinel_predictions, #goessgps10_ground_truth_dates, goessgps10_ground_truth_values, goessgps100_ground_truth_dates, goessgps100_ground_truth_values, 
                    goesxrs_ground_truth_dates, goesxrs_ground_truth_values, biosentinel_ground_truth_dates, biosentinel_ground_truth_values, test_file)

    test_plot_file = file_name + '.pdf'
    ylims = save_test_plot(context_dates, prediction_dates, training_prediction_window_end, #goessgps10_predictions, goessgps100_predictions, goesxrs_predictions, 
                        biosentinel_predictions, #goessgps10_ground_truth_dates, goessgps10_ground_truth_values, goessgps100_ground_truth_dates, goessgps100_ground_truth_values, 
                        goesxrs_ground_truth_dates, goesxrs_ground_truth_values, biosentinel_ground_truth_dates, biosentinel_ground_truth_values, test_plot_file, title=title)
    return ylims


def run_test_video(model, date_start, date_end, file_prefix, title_prefix, ylims, args):
    data_dir_sdo = os.path.join(args.data_dir, args.sdo_dir)
    data_dir_goes_xrs = os.path.join(args.data_dir, args.goes_xrs_file)
    data_dir_radlab = os.path.join(args.data_dir, args.radlab_file)

    full_start = date_start - datetime.timedelta(minutes=(model.context_window - 1) * args.delta_minutes)
    full_end = date_end
    dataset_goes_xrs = GOESXRS(data_dir_goes_xrs, date_start=full_start, date_end=date_end, random_data=args.xray_random_data)
    dataset_rad = RadLab(data_dir_radlab, instrument=args.rad_inst, date_start=full_start, date_end=date_end, random_data=args.rad_random_data)
    if isinstance(model, RadRecurrentWithSDO):
        dataset_sdo = SDOMLlite(data_dir_sdo, main_study_dir, date_start=full_start, date_end=date_end, random_data=args.sdo_random_data)
        full_start = max(dataset_sdo.date_start, dataset_goes_xrs.date_start, dataset_rad.date_start) # need to reassign because data availability may change the start date
        time_steps = int((full_end - full_start).total_seconds() / (args.delta_minutes * 60))
        dataset_sequences = Sequences([dataset_sdo, dataset_goes_xrs, dataset_rad], delta_minutes=args.delta_minutes, sequence_length=time_steps)
        if len(dataset_sequences) == 0:
            print('No data available for full sequence to generate video')
            return
        full_sequence = dataset_sequences[0]
        full_dates = [datetime.datetime.fromisoformat(d) for d in full_sequence[3]]
    elif isinstance(model, RadRecurrent):
        full_start = max(dataset_goes_xrs.date_start, dataset_rad.date_start)
        time_steps = int((full_end - full_start).total_seconds() / (args.delta_minutes * 60))
        dataset_sequences = Sequences([dataset_goes_xrs, dataset_rad], delta_minutes=args.delta_minutes, sequence_length=time_steps)
        if len(dataset_sequences) == 0:
            print('No data available for full sequence to generate video')
            return
        full_sequence = dataset_sequences[0]
        full_dates = [datetime.datetime.fromisoformat(d) for d in full_sequence[2]]
    else:
        raise ValueError('Unknown model type: {}'.format(model))

    if full_start != full_dates[0]:
        print('full start adjusted from {} to {} (due to data availability)'.format(full_start, full_dates[0]))
    full_start = full_dates[0]

    context_start = full_start
    context_end = context_start + datetime.timedelta(minutes=(model.context_window - 1) * args.delta_minutes)
    prediction_start = context_end
    prediction_end = full_end
    training_prediction_end = prediction_start + datetime.timedelta(minutes=model.prediction_window * args.delta_minutes)

    goesxrs_ground_truth_dates, goesxrs_ground_truth_values = dataset_goes_xrs.get_series(full_start, full_end, delta_minutes=args.delta_minutes)
    biosentinel_ground_truth_dates, biosentinel_ground_truth_values = dataset_rad.get_series(full_start, full_end, delta_minutes=args.delta_minutes)
    
    goesxrs_ground_truth_values = dataset_goes_xrs.unnormalize_data(goesxrs_ground_truth_values)
    biosentinel_ground_truth_values = dataset_rad.unnormalize_data(biosentinel_ground_truth_values)

    if isinstance(model, RadRecurrentWithSDO):
        fig, axs = plt.subplot_mosaic([['hmi_m', 'aia_0131', 'aia_0171', 'aia_0193', 'aia_0211', 'aia_1600'],
                                    ['biosentinel', 'biosentinel', 'biosentinel', 'biosentinel', 'biosentinel', 'biosentinel'],
                                    #['goessgps10', 'goessgps10', 'goessgps10', 'goessgps10', 'goessgps10', 'goessgps10'],
                                    #['goessgps100', 'goessgps100', 'goessgps100', 'goessgps100', 'goessgps100', 'goessgps100'],
                                    ['goesxrs', 'goesxrs', 'goesxrs', 'goesxrs', 'goesxrs', 'goesxrs']
                                    ], figsize=(20, 12.5), height_ratios=[1, 1, 1])
    elif isinstance(model, RadRecurrent):
        fig, axs = plt.subplot_mosaic([['biosentinel', 'biosentinel', 'biosentinel', 'biosentinel'],
                                    #['goessgps10', 'goessgps10', 'goessgps10', 'goessgps10'],
                                    #['goessgps100', 'goessgps100', 'goessgps100', 'goessgps100'],
                                    ['goesxrs', 'goesxrs', 'goesxrs', 'goesxrs']
                                    ], figsize=(20, 12.5), height_ratios=[1, 1])
    else:
        raise ValueError('Unknown model type: {}'.format(model))

    hours_locator = matplotlib.dates.HourLocator(interval=1)
    colors = {}
    colors['biosentinel'] = 'mediumblue'
    #colors['goessgps10'] = 'darkgreen'
    #colors['goessgps100'] = 'darkorange'
    colors['goesxrs'] = 'purple'
    colors['prediction'] = 'red'
    colors['prediction_mean'] = 'darkred'

    prediction_alpha = 0.08
    prediction_mean_alpha = 0.55
    prediction_secondary_alpha = prediction_alpha * 0.5
    prediction_secondary_mean_alpha =prediction_mean_alpha * 0.5

    channels = dataset_sdo.channels
    sdo_vmin = {}
    sdo_vmax = {}
    sdo_sample, _ = dataset_sdo[0]
    for i, c in enumerate(channels):
        if c == 'hmi_m':
            sdo_vmin[c], sdo_vmax[c] = -1500, 1500
        else:
            sdo_vmin[c], sdo_vmax[c] = np.percentile(dataset_sdo.unnormalize(sdo_sample[i], c), (0.2, 98))


    ims = {}
    for c in channels:
        cmap = sdo_cms[c]
        # cmap = 'viridis'    
        ax = axs[c]
        ax.set_title('SDOML-lite / {}'.format(c))
        ax.set_xticks([])
        ax.set_yticks([])
        ims[c] = ax.imshow(np.zeros([512,512]), vmin=sdo_vmin[c], vmax=sdo_vmax[c], cmap=cmap)

    ax = axs['biosentinel']
    # ax.set_title('Biosentinel BPD')
    ax.text(0.005, 0.96, 'BioSentinel absorbed dose rate', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    ax.set_ylabel('μGy/min')
    ax.yaxis.set_label_position("right")
    ax.plot(biosentinel_ground_truth_dates, biosentinel_ground_truth_values, color=colors['biosentinel'], alpha=0.75, label='Ground truth')
    ax.xaxis.set_minor_locator(hours_locator)
    ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
    ax.grid(color='lightgray', zorder=0, which='major')
    ax.set_xticklabels([])
    ax.set_yscale('log')    
    # ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    ims['biosentinel_context_start'] = ax.axvline(context_start, color=colors['prediction'], linestyle='--', linewidth=1) # Context start
    ims['biosentinel_prediction_start'] = ax.axvline(prediction_start, color=colors['prediction'], linestyle='-', linewidth=1.5) # Context end / Prediction start
    ims['biosentinel_training_prediction_end'] = ax.axvline(training_prediction_end, color=colors['prediction'], linestyle='--', linewidth=1) # Prediction end
    # ims['biosentinel_now_text'] = ax.text(prediction_start + datetime.timedelta(minutes=5), ylims['biosentinel'][0], 'Now',verticalalignment='bottom', horizontalalignment='left')
    # prediction plots
    ims['biosentinel_prediction_mean'] = ax.plot([], [], color=colors['prediction_mean'], alpha=prediction_mean_alpha, label='Prediction (mean)')[0]
    ims['biosentinel_prediction_std_upper'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_mean_alpha, label='Prediction (std dev)')[0]
    ims['biosentinel_prediction_std_lower'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_mean_alpha)[0]
    ims['biosentinel_prediction_mean_secondary'] = ax.plot([], [], color=colors['prediction_mean'], alpha=prediction_secondary_mean_alpha)[0]
    ims['biosentinel_prediction_std_secondary_upper'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_mean_alpha)[0]
    ims['biosentinel_prediction_std_secondary_lower'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_mean_alpha)[0]
    for i in range(args.num_samples):
        label = 'Prediction (samples)' if i == 0 else None
        ims['biosentinel_prediction_{}'.format(i)], = ax.plot([], [], color=colors['prediction'], alpha=prediction_alpha, label=label)
        ims['biosentinel_prediction_{}_secondary'.format(i)], = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_alpha)
    ax.legend(loc='upper right')
    ax.set_ylim(ylims['biosentinel'])

    # ax = axs['goessgps10']
    # # ax.set_title('GOES solar & galactic protons (>10MeV)')
    # ax.text(0.005, 0.96, 'GOES solar & galactic protons (>10 MeV)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    # ax.set_ylabel('part./(cm^2 s sr)')
    # ax.yaxis.set_label_position("right")
    # ax.plot(goessgps10_ground_truth_dates, goessgps10_ground_truth_values, color=colors['goessgps10'], alpha=0.75, label='Ground truth')
    # # ax.tick_params(rotation=45)
    # ax.set_xticks(axs['biosentinel'].get_xticks())
    # ax.set_xlim(axs['biosentinel'].get_xlim())
    # ax.xaxis.set_minor_locator(hours_locator)
    # ax.set_xticklabels([])
    # ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
    # ax.grid(color='lightgray', zorder=0, which='major')
    # ax.set_yscale('log')
    # ims['goessgps10_context_start'] = ax.axvline(context_start, color=colors['prediction'], linestyle='--', linewidth=1) # Context start
    # ims['goessgps10_prediction_start'] = ax.axvline(prediction_start, color=colors['prediction'], linestyle='-', linewidth=1.5) # Context end / Prediction start
    # ims['goessgps10_training_prediction_end'] = ax.axvline(training_prediction_end, color=colors['prediction'], linestyle='--', linewidth=1) # Prediction end
    # # ims['goessgps10_now_text'] = ax.text(prediction_start + datetime.timedelta(minutes=5), ylims['goessgps10'][0], 'Now',verticalalignment='bottom', horizontalalignment='left')
    # # prediction plots
    # ims['goessgps10_prediction_mean'] = ax.plot([], [], color=colors['prediction_mean'], alpha=prediction_mean_alpha)[0]
    # ims['goessgps10_prediction_std_upper'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_mean_alpha)[0]
    # ims['goessgps10_prediction_std_lower'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_mean_alpha)[0]
    # ims['goessgps10_prediction_mean_secondary'] = ax.plot([], [], color=colors['prediction_mean'], alpha=prediction_secondary_mean_alpha)[0]
    # ims['goessgps10_prediction_std_secondary_upper'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_mean_alpha)[0]
    # ims['goessgps10_prediction_std_secondary_lower'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_mean_alpha)[0]
    # for i in range(args.num_samples):
    #     ims['goessgps10_prediction_{}'.format(i)], = ax.plot([], [], color=colors['prediction'], alpha=prediction_alpha)
    #     ims['goessgps10_prediction_{}_secondary'.format(i)], = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_alpha)
    # ax.legend(loc='upper right')
    # ax.set_ylim(ylims['goessgps10'])

    # ax = axs['goessgps100']
    # # ax.set_title('GOES solar & galactic protons (>10MeV)')
    # ax.text(0.005, 0.96, 'GOES solar & galactic protons (>100 MeV)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    # ax.set_ylabel('part./(cm^2 s sr)')
    # ax.yaxis.set_label_position("right")
    # ax.plot(goessgps100_ground_truth_dates, goessgps100_ground_truth_values, color=colors['goessgps100'], alpha=0.75, label='Ground truth')
    # # ax.tick_params(rotation=45)
    # ax.set_xticks(axs['biosentinel'].get_xticks())
    # ax.set_xlim(axs['biosentinel'].get_xlim())
    # ax.xaxis.set_minor_locator(hours_locator)
    # ax.set_xticklabels([])
    # ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
    # ax.grid(color='lightgray', zorder=0, which='major')
    # ax.set_yscale('log')
    # ims['goessgps100_context_start'] = ax.axvline(context_start, color=colors['prediction'], linestyle='--', linewidth=1) # Context start
    # ims['goessgps100_prediction_start'] = ax.axvline(prediction_start, color=colors['prediction'], linestyle='-', linewidth=1.5) # Context end / Prediction start
    # ims['goessgps100_training_prediction_end'] = ax.axvline(training_prediction_end, color=colors['prediction'], linestyle='--', linewidth=1) # Prediction end
    # # ims['goessgps100_now_text'] = ax.text(prediction_start + datetime.timedelta(minutes=5), ylims['goessgps100'][0], 'Now',verticalalignment='bottom', horizontalalignment='left')
    # # prediction plots
    # ims['goessgps100_prediction_mean'] = ax.plot([], [], color=colors['prediction_mean'], alpha=prediction_mean_alpha)[0]
    # ims['goessgps100_prediction_std_upper'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_mean_alpha)[0]
    # ims['goessgps100_prediction_std_lower'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_mean_alpha)[0]
    # ims['goessgps100_prediction_mean_secondary'] = ax.plot([], [], color=colors['prediction_mean'], alpha=prediction_secondary_mean_alpha)[0]
    # ims['goessgps100_prediction_std_secondary_upper'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_mean_alpha)[0]
    # ims['goessgps100_prediction_std_secondary_lower'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_mean_alpha)[0]
    # for i in range(args.num_samples):
    #     ims['goessgps100_prediction_{}'.format(i)], = ax.plot([], [], color=colors['prediction'], alpha=prediction_alpha)
    #     ims['goessgps100_prediction_{}_secondary'.format(i)], = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_alpha)
    # ax.legend(loc='upper right')
    # ax.set_ylim(ylims['goessgps100'])

    ax = axs['goesxrs']
    # ax.set_title('GOES XRS')
    ax.text(0.005, 0.96, 'GOES X-ray flux', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    ax.set_ylabel('W/m^2')
    ax.yaxis.set_label_position("right")
    ax.plot(goesxrs_ground_truth_dates, goesxrs_ground_truth_values, color=colors['goesxrs'], alpha=0.75, label='Ground truth')
    # ax.tick_params(rotation=45)
    ax.set_xticks(axs['biosentinel'].get_xticks())
    ax.set_xlim(axs['biosentinel'].get_xlim())
    ax.xaxis.set_minor_locator(hours_locator)
    ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
    ax.grid(color='lightgray', zorder=0, which='major')
    ax.set_yscale('log')
    myFmt = matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    ims['goesxrs_context_start'] = ax.axvline(context_start, color=colors['prediction'], linestyle='--', linewidth=1) # Context start
    ims['goesxrs_prediction_start'] = ax.axvline(prediction_start, color=colors['prediction'], linestyle='-', linewidth=1.5) # Context end / Prediction start
    ims['goesxrs_training_prediction_end'] = ax.axvline(training_prediction_end, color=colors['prediction'], linestyle='--', linewidth=1) # Prediction end
    ims['goesxrs_now_text'] = ax.text(prediction_start + datetime.timedelta(minutes=5), ylims['goesxrs'][0], 'Now',verticalalignment='bottom', horizontalalignment='left')
    # prediction plots
    ims['goesxrs_prediction_mean'] = ax.plot([], [], color=colors['prediction_mean'], alpha=prediction_mean_alpha)[0]
    ims['goesxrs_prediction_std_upper'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_mean_alpha)[0]
    ims['goesxrs_prediction_std_lower'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_mean_alpha)[0]
    ims['goesxrs_prediction_mean_secondary'] = ax.plot([], [], color=colors['prediction_mean'], alpha=prediction_secondary_mean_alpha)[0]
    ims['goesxrs_prediction_std_secondary_upper'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_mean_alpha)[0]
    ims['goesxrs_prediction_std_secondary_lower'] = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_mean_alpha)[0]
    for i in range(args.num_samples):
        ims['goesxrs_prediction_{}'.format(i)], = ax.plot([], [], color=colors['prediction'], alpha=prediction_alpha)
        ims['goesxrs_prediction_{}_secondary'.format(i)], = ax.plot([], [], color=colors['prediction'], alpha=prediction_secondary_alpha)
    ax.legend(loc='upper right')
    ax.set_ylim(ylims['goesxrs'])

    title = plt.suptitle(title_prefix + ' ' + str(prediction_start), y=0.995)

    # num_frames = int(((full_end - prediction_start).total_seconds() / 60) / args.delta_minutes) + 1
    num_frames = len(full_dates) - model.context_window

    with tqdm(total=num_frames) as pbar:
        def run(i):
            context_start = i # sliding context
            # context_start = 0 # full context
            context_end = i + model.context_window - 1

            # context_window = context_end - context_start + 1
            prediction_start = context_end
            # prediction_end = len(full_dates) - 1
            prediction_window = num_frames - i + 1        

            context_start_date = full_dates[context_start]
            # context_end_date = full_dates[context_end]
            prediction_start_date = full_dates[prediction_start]
            # prediction_end_date = full_dates[prediction_end]
            training_prediction_end_date = prediction_start_date + datetime.timedelta(minutes=model.prediction_window * args.delta_minutes)

            if isinstance(model, RadRecurrentWithSDO):
                context_sdo = full_sequence[0][context_start:context_end+1].to(args.device)
                context_sdo_batch = context_sdo.unsqueeze(0)
                # context_goessgps10 = full_sequence[1][context_start:context_end+1].unsqueeze(1).to(args.device)
                # context_goessgps100 = full_sequence[2][context_start:context_end+1].unsqueeze(1).to(args.device)
                context_goesxrs = full_sequence[1][context_start:context_end+1].unsqueeze(1).to(args.device)
                context_rad = full_sequence[1][context_start:context_end+1].unsqueeze(1).to(args.device)
                context_data = torch.cat([context_goesxrs, context_rad], dim=1)
                context_data_batch = context_data.unsqueeze(0)
                prediction_batch = model.predict(context_sdo_batch, context_data_batch, prediction_window, num_samples=args.num_samples).detach()
            elif isinstance(model, RadRecurrent):
                context_goessgps10 = full_sequence[0][context_start:context_end+1].unsqueeze(1).to(args.device)
                context_goessgps100 = full_sequence[1][context_start:context_end+1].unsqueeze(1).to(args.device)
                context_goesxrs = full_sequence[2][context_start:context_end+1].unsqueeze(1).to(args.device)
                context_rad = full_sequence[3][context_start:context_end+1].unsqueeze(1).to(args.device)
                context = torch.cat([context_goessgps10, context_goessgps100, context_goesxrs, context_rad], dim=1)
                context_batch = context.unsqueeze(0).repeat(args.num_samples, 1, 1)
                prediction_batch = model.predict(context_batch, prediction_window).detach()
            else:
                raise ValueError('Unknown model type: {}'.format(model))

            # goessgps10_predictions = prediction_batch[:, :, 0]
            # goessgps100_predictions = prediction_batch[:, :, 1]
            # goesxrs_predictions = prediction_batch[:, :, 0]
            biosentinel_predictions = prediction_batch[:, :, 0]
            # goessgps10_predictions = dataset_goes_sgps10.unnormalize_data(goessgps10_predictions).cpu().numpy()
            # goessgps100_predictions = dataset_goes_sgps100.unnormalize_data(goessgps100_predictions).cpu().numpy()
            # goesxrs_predictions = dataset_goes_xrs.unnormalize_data(goesxrs_predictions).cpu().numpy()
            biosentinel_predictions = dataset_rad.unnormalize_data(biosentinel_predictions).cpu().numpy()
            prediction_dates = [prediction_start_date + datetime.timedelta(minutes=i*args.delta_minutes) for i in range(prediction_window + 1)]

            title.set_text(title_prefix + ' ' + str(prediction_start_date))
            ims['biosentinel_context_start'].set_xdata([context_start_date, context_start_date])
            ims['biosentinel_prediction_start'].set_xdata([prediction_start_date, prediction_start_date])
            ims['biosentinel_training_prediction_end'].set_xdata([training_prediction_end_date, training_prediction_end_date])
            # ims['biosentinel_now_text'].set_position((prediction_start_date + datetime.timedelta(minutes=5), ylims['biosentinel'][0]))

            # ims['goessgps10_context_start'].set_xdata([context_start_date, context_start_date])
            # ims['goessgps10_prediction_start'].set_xdata([prediction_start_date, prediction_start_date])
            # ims['goessgps10_training_prediction_end'].set_xdata([training_prediction_end_date, training_prediction_end_date])
            # # ims['goessgps10_now_text'].set_position((prediction_start_date + datetime.timedelta(minutes=5), ylims['goessgps10'][0]))

            # ims['goessgps100_context_start'].set_xdata([context_start_date, context_start_date])
            # ims['goessgps100_prediction_start'].set_xdata([prediction_start_date, prediction_start_date])
            # ims['goessgps100_training_prediction_end'].set_xdata([training_prediction_end_date, training_prediction_end_date])
            # # ims['goessgps100_now_text'].set_position((prediction_start_date + datetime.timedelta(minutes=5), ylims['goessgps100'][0]))

            ims['goesxrs_context_start'].set_xdata([context_start_date, context_start_date])
            ims['goesxrs_prediction_start'].set_xdata([prediction_start_date, prediction_start_date])
            ims['goesxrs_training_prediction_end'].set_xdata([training_prediction_end_date, training_prediction_end_date])
            ims['goesxrs_now_text'].set_position((prediction_start_date + datetime.timedelta(minutes=5), ylims['goesxrs'][0]))

            if isinstance(model, RadRecurrentWithSDO):
                sdo_data = context_sdo[-1]
                for i, c in enumerate(channels):
                    if sdo_data is None:
                        # ims[c].set_data(np.zeros([512,512]))
                        pass
                    else:
                        ims[c].set_data(dataset_sdo.unnormalize(sdo_data[i].cpu().numpy(), c))

            prediction_dates_primary = prediction_dates[:model.prediction_window+1]
            prediction_dates_secondary = prediction_dates[model.prediction_window:]
            biosentinel_predictions_primary = biosentinel_predictions[:, :model.prediction_window+1]
            biosentinel_predictions_secondary = biosentinel_predictions[:, model.prediction_window:]
            # goessgps10_predictions_primary = goessgps10_predictions[:, :model.prediction_window+1]
            # goessgps10_predictions_secondary = goessgps10_predictions[:, model.prediction_window:]
            # goessgps100_predictions_primary = goessgps100_predictions[:, :model.prediction_window+1]
            # goessgps100_predictions_secondary = goessgps100_predictions[:, model.prediction_window:]
            # goesxrs_predictions_primary = goesxrs_predictions[:, :model.prediction_window+1]
            # goesxrs_predictions_secondary = goesxrs_predictions[:, model.prediction_window:]

            ims['biosentinel_prediction_mean'].set_data(prediction_dates_primary, np.mean(biosentinel_predictions_primary, axis=0))
            # ims['goessgps10_prediction_mean'].set_data(prediction_dates_primary, np.mean(goessgps10_predictions_primary, axis=0))
            # ims['goessgps100_prediction_mean'].set_data(prediction_dates_primary, np.mean(goessgps100_predictions_primary, axis=0))
            # ims['goesxrs_prediction_mean'].set_data(prediction_dates_primary, np.mean(goesxrs_predictions_primary, axis=0))

            ims['biosentinel_prediction_std_upper'].set_data(prediction_dates_primary, np.mean(biosentinel_predictions_primary, axis=0) + np.std(biosentinel_predictions_primary, axis=0))
            ims['biosentinel_prediction_std_lower'].set_data(prediction_dates_primary, np.mean(biosentinel_predictions_primary, axis=0) - np.std(biosentinel_predictions_primary, axis=0))
            # ims['goessgps10_prediction_std_lower'].set_data(prediction_dates_primary, np.mean(goessgps10_predictions_primary, axis=0) - np.std(goessgps10_predictions_primary, axis=0))
            # ims['goessgps10_prediction_std_upper'].set_data(prediction_dates_primary, np.mean(goessgps10_predictions_primary, axis=0) + np.std(goessgps10_predictions_primary, axis=0))
            # ims['goessgps100_prediction_std_lower'].set_data(prediction_dates_primary, np.mean(goessgps100_predictions_primary, axis=0) - np.std(goessgps100_predictions_primary, axis=0))
            # ims['goessgps100_prediction_std_upper'].set_data(prediction_dates_primary, np.mean(goessgps100_predictions_primary, axis=0) + np.std(goessgps100_predictions_primary, axis=0))
            # ims['goesxrs_prediction_std_lower'].set_data(prediction_dates_primary, np.mean(goesxrs_predictions_primary, axis=0) - np.std(goesxrs_predictions_primary, axis=0))
            # ims['goesxrs_prediction_std_upper'].set_data(prediction_dates_primary, np.mean(goesxrs_predictions_primary, axis=0) + np.std(goesxrs_predictions_primary, axis=0))

            ims['biosentinel_prediction_mean_secondary'].set_data(prediction_dates_secondary, np.mean(biosentinel_predictions_secondary, axis=0))
            # ims['goessgps10_prediction_mean_secondary'].set_data(prediction_dates_secondary, np.mean(goessgps10_predictions_secondary, axis=0))
            # ims['goessgps100_prediction_mean_secondary'].set_data(prediction_dates_secondary, np.mean(goessgps100_predictions_secondary, axis=0))
            # ims['goesxrs_prediction_mean_secondary'].set_data(prediction_dates_secondary, np.mean(goesxrs_predictions_secondary, axis=0))

            ims['biosentinel_prediction_std_secondary_upper'].set_data(prediction_dates_secondary, np.mean(biosentinel_predictions_secondary, axis=0) + np.std(biosentinel_predictions_secondary, axis=0))
            ims['biosentinel_prediction_std_secondary_lower'].set_data(prediction_dates_secondary, np.mean(biosentinel_predictions_secondary, axis=0) - np.std(biosentinel_predictions_secondary, axis=0))
            # ims['goessgps10_prediction_std_secondary_lower'].set_data(prediction_dates_secondary, np.mean(goessgps10_predictions_secondary, axis=0) - np.std(goessgps10_predictions_secondary, axis=0))
            # ims['goessgps10_prediction_std_secondary_upper'].set_data(prediction_dates_secondary, np.mean(goessgps10_predictions_secondary, axis=0) + np.std(goessgps10_predictions_secondary, axis=0))
            # ims['goessgps100_prediction_std_secondary_lower'].set_data(prediction_dates_secondary, np.mean(goessgps100_predictions_secondary, axis=0) - np.std(goessgps100_predictions_secondary, axis=0))
            # ims['goessgps100_prediction_std_secondary_upper'].set_data(prediction_dates_secondary, np.mean(goessgps100_predictions_secondary, axis=0) + np.std(goessgps100_predictions_secondary, axis=0))
            # ims['goesxrs_prediction_std_secondary_lower'].set_data(prediction_dates_secondary, np.mean(goesxrs_predictions_secondary, axis=0) - np.std(goesxrs_predictions_secondary, axis=0))
            # ims['goesxrs_prediction_std_secondary_upper'].set_data(prediction_dates_secondary, np.mean(goesxrs_predictions_secondary, axis=0) + np.std(goesxrs_predictions_secondary, axis=0))

            for i in range(args.num_samples):
                ims['biosentinel_prediction_{}'.format(i)].set_data(prediction_dates_primary, biosentinel_predictions_primary[i])
                # ims['goessgps10_prediction_{}'.format(i)].set_data(prediction_dates_primary, goessgps10_predictions_primary[i])
                # ims['goessgps100_prediction_{}'.format(i)].set_data(prediction_dates_primary, goessgps100_predictions_primary[i])
                # ims['goesxrs_prediction_{}'.format(i)].set_data(prediction_dates_primary, goesxrs_predictions_primary[i])

                ims['biosentinel_prediction_{}_secondary'.format(i)].set_data(prediction_dates_secondary, biosentinel_predictions_secondary[i])
                # ims['goessgps10_prediction_{}_secondary'.format(i)].set_data(prediction_dates_secondary, goessgps10_predictions_secondary[i])
                # ims['goessgps100_prediction_{}_secondary'.format(i)].set_data(prediction_dates_secondary, goessgps100_predictions_secondary[i])
                # ims['goesxrs_prediction_{}_secondary'.format(i)].set_data(prediction_dates_secondary, goesxrs_predictions_secondary[i])

            pbar.set_description('Frame {}'.format(prediction_start_date))
            pbar.update(1)

        # plt.tight_layout()
        plt.tight_layout(rect=[0, 0, 1, 1.005])
        anim = animation.FuncAnimation(fig, run, interval=300, frames=num_frames)
        
        file_name = os.path.join(args.target_dir, file_prefix)
        file_name_mp4 = file_name + '.mp4'
        print('Saving video to {}'.format(file_name_mp4))
        writer_mp4 = animation.FFMpegWriter(fps=15)
        anim.save(file_name_mp4, writer=writer_mp4)


def save_loss_plot(train_losses, valid_losses, plot_file):
    print('Saving plot to {}'.format(plot_file))
    plt.figure(figsize=(12, 6))
    plt.plot(*zip(*train_losses), label='Training')
    plt.plot(*zip(*valid_losses), label='Validation')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(color='#f0f0f0', zorder=0)
    plt.tight_layout()
    plt.savefig(plot_file)    


def save_test_numpy(model, date_start, date_end, main_study_dir, file_prefix, args):
    data_dir_sdo = os.path.join(args.data_dir, args.sdo_dir)
    data_dir_goes_xrs = os.path.join(args.data_dir, args.goes_xrs_file)
    data_dir_radlab = os.path.join(args.data_dir, args.radlab_file)

    # Different compared to run_test_video: 
    # full_start = initial test time = beginning of first context window
    # full_end goes beyond test interval
    full_start = date_start # - datetime.timedelta(minutes=(model.context_window - 1) * args.delta_minutes)
    prediction_window = model.prediction_window * args.multiples_prediction_window
    full_end = date_end + datetime.timedelta(minutes=prediction_window * args.delta_minutes)

    dataset_goes_xrs = GOESXRS(data_dir_goes_xrs, date_start=full_start, date_end=full_end, rewind_minutes=args.delta_minutes, random_data=args.xray_random_data)
    dataset_rad = RadLab(data_dir_radlab, instrument=args.rad_inst, date_start=full_start, date_end=full_end, rewind_minutes=args.delta_minutes, random_data=args.rad_random_data)
    if isinstance(model, RadRecurrentWithSDO):
        dataset_sdo = SDOMLlite(data_dir_sdo, main_study_dir, date_start=full_start, date_end=full_end, random_data=args.sdo_random_data)
        full_start = max(dataset_sdo.date_start, 
                        dataset_goes_xrs.date_start, dataset_rad.date_start) # need to reassign because data availability may change the start date
        time_steps = int((full_end - full_start).total_seconds() / (args.delta_minutes * 60))
        dataset_sequences = Sequences([dataset_sdo, 
                                    dataset_goes_xrs, dataset_rad], delta_minutes=args.delta_minutes, sequence_length=time_steps)
        if len(dataset_sequences) == 0:
            print('No data available for full sequence to generate predictions')
            return
        full_sequence = dataset_sequences[0]
        full_dates = [datetime.datetime.fromisoformat(d) for d in full_sequence[3]]
    elif isinstance(model, RadRecurrentWithSDOCore):
        dataset_sdo = SDOCore(data_dir_sdo, date_start=full_start, date_end=full_end, random_data=args.sdo_random_data)
        full_start = max(dataset_sdo.date_start,
                        dataset_goes_xrs.date_start, dataset_rad.date_start)
        time_steps = int((full_end - full_start).total_seconds() / (args.delta_minutes * 60))
        dataset_sequences = Sequences([dataset_sdo,
                                    dataset_goes_xrs, dataset_rad], delta_minutes=args.delta_minutes, sequence_length=time_steps)
        if len(dataset_sequences) == 0:
            print('No data available for full sequence to generate predictions')
            return
        full_sequence = dataset_sequences[0]
        full_dates = [datetime.datetime.fromisoformat(d) for d in full_sequence[3]]
    elif isinstance(model, RadRecurrent):
        full_start = max(dataset_goes_xrs.date_start, dataset_rad.date_start)
        time_steps = int((full_end - full_start).total_seconds() / (args.delta_minutes * 60))
        dataset_sequences = Sequences([dataset_goes_xrs, dataset_rad], delta_minutes=args.delta_minutes, sequence_length=time_steps)
        if len(dataset_sequences) == 0:
            print('No data available for full sequence to generate predictions')
            return
        full_sequence = dataset_sequences[0]
        full_dates = [datetime.datetime.fromisoformat(d) for d in full_sequence[2]]
    else:
        raise ValueError('Unknown model type: {}'.format(model))

    if full_start != full_dates[0]:
        print('full start adjusted from {} to {} (due to data availability)'.format(full_start, full_dates[0]))
    full_start = full_dates[0]

    
    # get the ground truth within the test event
    # goessgps10_ground_truth_dates, goessgps10_ground_truth_values = dataset_goes_sgps10.get_series(full_start, full_end, delta_minutes=args.delta_minutes)
    # goessgps100_ground_truth_dates, goessgps100_ground_truth_values = dataset_goes_sgps100.get_series(full_start, full_end, delta_minutes=args.delta_minutes)
    goesxrs_ground_truth_dates, goesxrs_ground_truth_values = dataset_goes_xrs.get_series(full_start, full_end, delta_minutes=args.delta_minutes)
    biosentinel_ground_truth_dates, biosentinel_ground_truth_values = dataset_rad.get_series(full_start, full_end, delta_minutes=args.delta_minutes)
    # goessgps10_ground_truth_values = dataset_goes_sgps10.unnormalize_data(goessgps10_ground_truth_values)
    # goessgps100_ground_truth_values = dataset_goes_sgps100.unnormalize_data(goessgps100_ground_truth_values)
    goesxrs_ground_truth_values = dataset_goes_xrs.unnormalize_data(goesxrs_ground_truth_values)
    biosentinel_ground_truth_values = dataset_rad.unnormalize_data(biosentinel_ground_truth_values)

    # number of datapoints (now-times)
    num_frames = len(full_dates) - model.context_window - prediction_window

    current_now_day = full_dates[model.context_window - 1]
    biosentinel_p = []
    dates_p = []
    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            context_start = i # sliding context
            context_end = i + model.context_window - 1
            prediction_start = context_end

            context_start_date = full_dates[context_start]
            prediction_start_date = full_dates[prediction_start]

            if not prediction_start_date.day == current_now_day.day:
                file_rad = main_study_dir+'/test/saved_predictions/{}-rad_pred-{}-{}wprediction.npy'.format(file_prefix,current_now_day.strftime('%Y%m%d'),args.multiples_prediction_window)
                print('Saving into...',file_rad)
                biosentinel_p = np.array(biosentinel_p)
                np.save(file_rad,biosentinel_p)

                file_dates = main_study_dir+'/test/saved_predictions/{}-dates-{}-{}wprediction.npy'.format(file_prefix,current_now_day.strftime('%Y%m%d'),args.multiples_prediction_window)
                print('Saving into...',file_dates)
                dates_p = np.array(dates_p)
                np.save(file_dates,dates_p)

                dates_p = []
                biosentinel_p = []
                current_now_day = full_dates[prediction_start]

            if isinstance(model, RadRecurrentWithSDO):
                context_sdo = full_sequence[0][context_start:context_end+1].to(args.device)
                context_sdo_batch = context_sdo.unsqueeze(0)
                context_goesxrs = full_sequence[1][context_start:context_end+1].unsqueeze(1).to(args.device)
                context_rad = full_sequence[2][context_start:context_end+1].unsqueeze(1).to(args.device)
                context_data = torch.cat([context_goesxrs, context_rad], dim=1)
                context_data_batch = context_data.unsqueeze(0)
                prediction_batch = model.predict(context_sdo_batch, context_data_batch, prediction_window, num_samples=args.num_samples).detach()
            elif isinstance(model, RadRecurrent):
                context_goesxrs = full_sequence[1][context_start:context_end+1].unsqueeze(1).to(args.device)
                context_rad = full_sequence[2][context_start:context_end+1].unsqueeze(1).to(args.device)
                context = torch.cat([context_goesxrs, context_rad], dim=1)
                context_batch = context.unsqueeze(0).repeat(args.num_samples, 1, 1)
                prediction_batch = model.predict(context_batch, prediction_window).detach()
            else:
                raise ValueError('Unknown model type: {}'.format(model))

            #goesxrs_predictions = prediction_batch[:, :, 0]
            biosentinel_predictions = prediction_batch[:, :, 1]
            #goesxrs_predictions = dataset_goes_xrs.unnormalize_data(goesxrs_predictions).cpu().numpy()
            biosentinel_predictions = dataset_rad.unnormalize_data(biosentinel_predictions).cpu().numpy()
            prediction_dates = [prediction_start_date + datetime.timedelta(minutes=i*args.delta_minutes) for i in range(prediction_window + 1)]

            """#  divide predictions in within prediction window and beyond prediction window
            prediction_dates_primary = prediction_dates[:model.prediction_window+1]
            prediction_dates_secondary = prediction_dates[model.prediction_window+1:]
            biosentinel_predictions_primary = biosentinel_predictions[:, :model.prediction_window+1]
            biosentinel_predictions_secondary = biosentinel_predictions[:, model.prediction_window+1:]
            goessgps10_predictions_primary = goessgps10_predictions[:, :model.prediction_window+1]
            goessgps10_predictions_secondary = goessgps10_predictions[:, model.prediction_window+1:]
            goessgps100_predictions_primary = goessgps100_predictions[:, :model.prediction_window+1]
            goessgps100_predictions_secondary = goessgps100_predictions[:, model.prediction_window+1:]
            goesxrs_predictions_primary = goesxrs_predictions[:, :model.prediction_window+1]
            goesxrs_predictions_secondary = goesxrs_predictions[:, model.prediction_window+1:]"""

            pbar.set_description('Frame {}'.format(prediction_start_date))
            pbar.update(1)
    
            biosentinel_p.append(biosentinel_predictions)
            #goesxrs_p.append(goesxrs_predictions)
            dates_p.append(np.reshape(np.tile(prediction_dates,args.num_samples),(args.num_samples,len(prediction_dates))))
            
    
def main():
    description = 'FDL-X 2024, Radiation Team, preliminary machine learning experiments'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to store results')#/home/emassara/2024-hl-radiation-ml/results
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory with datasets')#/mnt/disks/hl-dosi-datasets/data/
    #parser.add_argument('--sdo_dir', type=str, default='sdoml-lite', help='SDOML-lite directory')
    parser.add_argument('--solar_dataset', type=str, choices=['SDOMLlite', 'SDOCore'], default='SDOMLlite', help='Solar dataset type')
    parser.add_argument('--rad_inst', type=str, choices=['CRaTER-D1D2'], default='CRaTER-D1D2', help='Radiation instrument')
    parser.add_argument('--sdo_random_data', action='store_true', help='Use fake SDO data (for ablation study)')
    parser.add_argument('--xray_random_data', action='store_true', help='Use fake xray data (for ablation study)')
    parser.add_argument('--rad_random_data', action='store_true', help='Use fake radiation data (for ablation study)')
    # parser.add_argument('--sdo_only_context', action='store_true', help='Use only SDO data for context') ## Taking a different approach (i.e. passing random_data) for ablation study instead of using this flag
    parser.add_argument('--radlab_file', type=str, default='radlab-private/RadLab-20240625-duck-corrected.db', help='RadLab file') #USE CORRECTED once Rutuja updates it
    parser.add_argument('--goes_xrs_file', type=str, default='goes/goes-xrs.csv', help='GOES XRS file')
    parser.add_argument('--goes_sgps_file', type=str, default='goes/goes-sgps.csv', help='GOES SGPS file')
    parser.add_argument('--context_window', type=int, default=40, help='Context window')
    parser.add_argument('--prediction_window', type=int, default=40, help='Prediction window')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples for MC dropout inference')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Delta minutes') # maybe set it to the cadence of solar images (12 for SDOcore, 15 for SDOMLlite)
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--seed', type=int, default=0, help='Random number generator seed')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--valid_proportion', type=float, default=0.05, help='Validation frequency in iterations')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--lstm_depth', type=int, default=2, help='LSTM depth')
    parser.add_argument('--model_type', type=str, choices=['RadRecurrent', 'RadRecurrentWithSDO','RadRecurrentWithSDOCore'], default='RadRecurrentWithSDO', help='Model type')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='Mode', required=True)
    parser.add_argument('--date_start', type=str, default='2017-02-07T00:00:00', help='Start date') #default='2022-11-16T11:00:00'
    parser.add_argument('--date_end', type=str, default='2024-05-31T23:59:59', help='End date')     #default='2024-05-14T09:15:00'
    parser.add_argument('--test_event_id', nargs='+', default=['test08','test09','test10','test11','test12','test13','test14','test15'], help='Test event IDs')
    parser.add_argument('--valid_event_id', nargs='+', default=['valid08','valid09','valid10','valid11','valid12','valid13','valid14','valid15'], help='Validation event IDs')
    # parser.add_argument('--test_seen_event_id', nargs='+', default=['biosentinel04', 'biosentinel15', 'biosentinel18'], help='Test event IDs seen during training')
    # parser.add_argument('--test_event_id', nargs='+', default=['biosentinel06'], help='Test event IDs')
    # parser.add_argument('--test_seen_event_id', nargs='+', default=None, help='Test event IDs seen during training')

    parser.add_argument('--model_file', type=str, help='Model file')
    parser.add_argument('--multiples_prediction_window', type=int, default=1, help='multiples_prediction_window')

    args = parser.parse_args()
    if args.solar_dataset == 'SDOMLlite':
        args.sdo_dir = 'sdoml-lite'
    elif args.solar_dataset == 'SDOCore':
        args.sdo_dir = 'sdocore'

    # make sure the target directory exists
    os.makedirs(args.target_dir, exist_ok=True)

    ## Create the ablation study ids
    if not args.sdo_random_data and not args.xray_random_data and not args.rad_random_data:
        study_id = "study--solar-rad-xray_to_rad" ## This is the "max" study
    elif not args.sdo_random_data and not args.xray_random_data and args.rad_random_data:
        study_id = "study--solar-xray_to_rad"
    elif not args.sdo_random_data and args.xray_random_data and args.rad_random_data:
        study_id = "study--solar_to_rad"
    elif args.sdo_random_data and not args.xray_random_data and args.rad_random_data:
        study_id = "study--xray_to_rad"
    
    ## Create the study directory with the folder structure as decided: /home/username/2024-hl-radiation-ml/results/solar-inst[SDOCORE]/rad-inst[CRaTER]/study–solar-rad-xray_to_rad
    main_study_dir = args.target_dir+f"/solar-dataset[{args.solar_dataset}]/rad-inst[{args.rad_inst}]/{study_id}/{args.date_start}-{args.date_end}"
    os.makedirs(main_study_dir, exist_ok=True)
    
    ## Create the various subdirectories to hold results, plots, etc for each study
    subdirs = ['saved_model', 'train/plots', 'train/logs', 'test/plots', 'test/logs', 'test/saved_predictions', 'train/loss']
    for subdir in subdirs:
        os.makedirs(main_study_dir+'/'+subdir, exist_ok=True)

    ## Create the log file for the study depending on the run.py mode
    log_file = os.path.join(main_study_dir, f'{args.mode}/log.txt')

    with Tee(log_file):
        print(description)    
        print('Log file: {}'.format(log_file))
        start_time = datetime.datetime.now()
        print('Start time: {}'.format(start_time))
        print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
        print('Config:')
        pprint.pprint(vars(args), depth=2, width=50)

        seed(args.seed)
        device = torch.device(args.device)


        data_dir_sdo = os.path.join(args.data_dir, args.sdo_dir)
        #data_dir_goes_sgps = os.path.join(args.data_dir, args.goes_sgps_file)
        data_dir_goes_xrs = os.path.join(args.data_dir, args.goes_xrs_file)
        data_dir_radlab = os.path.join(args.data_dir, args.radlab_file)

        sys.stdout.flush()
        if args.mode == 'train':
            ## Creating Dataloaders...
            print('\n*** Training mode\n')

            training_sequence_length = args.context_window + args.prediction_window

            print('Processing excluded dates')
            if args.model_type in ['RadRecurrentWithSDO', 'RadRecurrentWithSDOCore']:
                datasets_sdo_valid = []
                datasets_sdo_test = []
            datasets_goes_xrs_valid = []
            datasets_rad_valid = []
            datasets_goes_xrs_test = []
            datasets_rad_test = []

            date_exclusions = []
            # Validation set
            if args.valid_event_id is not None:
                for event_id in args.valid_event_id:
                    print('Excluding event: {}'.format(event_id))
                    if event_id not in EventCatalog:
                        raise ValueError('Event ID not found in events: {}'.format(event_id))
                    exclusion_start, exclusion_end = EventCatalog[event_id]
                    exclusion_start = datetime.datetime.fromisoformat(exclusion_start)
                    exclusion_end = datetime.datetime.fromisoformat(exclusion_end)
                    date_exclusions.append((exclusion_start, exclusion_end))

                    if args.model_type == 'RadRecurrentWithSDO':
                        datasets_sdo_valid.append(SDOMLlite(data_dir_sdo, main_study_dir, date_start=exclusion_start, date_end=exclusion_end, random_data=args.sdo_random_data))
                    elif args.model_type == 'RadRecurrentWithSDOCore':
                        datasets_sdo_valid.append(SDOCore(data_dir_sdo, date_start=exclusion_start, date_end=exclusion_end, random_data=args.sdo_random_data))
                    datasets_goes_xrs_valid.append(GOESXRS(data_dir_goes_xrs, date_start=exclusion_start, date_end=exclusion_end,rewind_minutes=args.delta_minutes, random_data=args.xray_random_data))
                    datasets_rad_valid.append(RadLab(data_dir_radlab, instrument=args.rad_inst, date_start=exclusion_start, date_end=exclusion_end, rewind_minutes=args.delta_minutes, random_data=args.rad_random_data))
            if args.model_type in ['RadRecurrentWithSDO', 'RadRecurrentWithSDOCore']:
                dataset_sdo_valid = UnionDataset(datasets_sdo_valid)
            dataset_goes_xrs_valid = UnionDataset(datasets_goes_xrs_valid)
            dataset_rad_valid = UnionDataset(datasets_rad_valid)
            if args.model_type in ['RadRecurrentWithSDO', 'RadRecurrentWithSDOCore']:
                dataset_sequences_valid = Sequences([dataset_sdo_valid, 
                                                    dataset_goes_xrs_valid, dataset_rad_valid], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
            else:
                dataset_sequences_valid = Sequences([dataset_goes_xrs_valid, dataset_rad_valid], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)

            # Test set
            if args.test_event_id is not None:
                for event_id in args.test_event_id:
                    print('Excluding event: {}'.format(event_id))
                    if event_id not in EventCatalog:
                        raise ValueError('Event ID not found in events: {}'.format(event_id))
                    exclusion_start, exclusion_end = EventCatalog[event_id]
                    exclusion_start = datetime.datetime.fromisoformat(exclusion_start)
                    exclusion_end = datetime.datetime.fromisoformat(exclusion_end)
                    date_exclusions.append((exclusion_start, exclusion_end))

                    if args.model_type == 'RadRecurrentWithSDO':
                        datasets_sdo_test.append(SDOMLlite(data_dir_sdo, main_study_dir, date_start=exclusion_start, date_end=exclusion_end, random_data=args.sdo_random_data))
                    elif args.model_type == 'RadRecurrentWithSDOCore': 
                        datasets_sdo_test.append(SDOCore(data_dir_sdo, date_start=exclusion_start, date_end=exclusion_end, random_data=args.sdo_random_data))
                    datasets_goes_xrs_test.append(GOESXRS(data_dir_goes_xrs, date_start=exclusion_start, date_end=exclusion_end,rewind_minutes=args.delta_minutes, random_data=args.xray_random_data))
                    datasets_rad_test.append(RadLab(data_dir_radlab, instrument=args.rad_inst, date_start=exclusion_start, date_end=exclusion_end,rewind_minutes=args.delta_minutes, random_data=args.rad_random_data))
            if args.model_type in ['RadRecurrentWithSDO', 'RadRecurrentWithSDOCore']:
                dataset_sdo_test = UnionDataset(datasets_sdo_test)
            dataset_goes_xrs_test = UnionDataset(datasets_goes_xrs_test)
            dataset_rad_test = UnionDataset(datasets_rad_test)
            if args.model_type in ['RadRecurrentWithSDO', 'RadRecurrentWithSDOCore']:
                dataset_sequences_test = Sequences([dataset_sdo_test, 
                                                    dataset_goes_xrs_test, dataset_rad_test], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
            else:
                dataset_sequences_test = Sequences([dataset_goes_xrs_test, dataset_rad_test], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)


            # Training set
            if args.model_type == 'RadRecurrentWithSDO':
                dataset_sdo = SDOMLlite(data_dir_sdo, main_study_dir, date_start=args.date_start, date_end=args.date_end, date_exclusions=date_exclusions, random_data=args.sdo_random_data)

            elif args.model_type == 'RadRecurrentWithSDOCore':
                dataset_sdo = SDOCore(data_dir_sdo, date_start=args.date_start, date_end=args.date_end, date_exclusions=date_exclusions, random_data=args.sdo_random_data)

            dataset_goes_xrs = GOESXRS(data_dir_goes_xrs, date_start=args.date_start, date_end=args.date_end, date_exclusions=date_exclusions, rewind_minutes=args.delta_minutes, random_data=args.xray_random_data)
            dataset_rad = RadLab(data_dir_radlab, instrument=args.rad_inst, date_start=args.date_start, date_end=args.date_end, date_exclusions=date_exclusions, rewind_minutes=args.delta_minutes, random_data=args.rad_random_data)
            if args.model_type in ['RadRecurrentWithSDO', 'RadRecurrentWithSDOCore']:
                dataset_sequences_train = Sequences([dataset_sdo, 
                                                    dataset_goes_xrs, dataset_rad], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
            else:
                dataset_sequences_train = Sequences([dataset_goes_xrs, dataset_rad], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)

            print('\nTrain size: {:,}'.format(len(dataset_sequences_train)))
            print('Valid size: {:,}'.format(len(dataset_sequences_valid)))
            print('Test size: {:,}'.format(len(dataset_sequences_test)))

            train_loader = DataLoader(dataset_sequences_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            valid_loader = DataLoader(dataset_sequences_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            #test_loader = DataLoader(dataset_sequences_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            ## Creating model...            
            # check if a previous training run exists in the target directory, if so, find the latest model file saved, resume training from there by loading the model instead of creating a new one
            model_files = []
            for entry in os.scandir('{}/saved_model'.format(main_study_dir)):
                if entry.name.startswith('epoch'):
                    model_files.append(entry.name)
            if len(model_files) > 0:
                model_files.sort()
                model_file = model_files[-1]
                print('Resuming training from model file: {}'.format(model_file))
                model, optimizer, epoch, iteration, train_losses, valid_losses = load_model(model_file, device)
                # model_data_dim = model.data_dim
                # model_lstm_dim = model.lstm_dim
                # model_lstm_depth = model.lstm_depth
                # model_dropout = model.dropout
                # model_context_window = model.context_window
                # model_prediction_window = model.prediction_window
                epoch_start = epoch + 1
                iteration = iteration + 1
                print('Next epoch    : {:,}'.format(epoch_start+1))
                print('Next iteration: {:,}'.format(iteration+1))
            else:
                print('Creating new model')
                if args.model_type == 'RadRecurrent':
                    model_data_dim = 2
                    model_lstm_dim = 1024
                    model_lstm_depth = args.lstm_depth
                    model_dropout = 0.2
                    model_context_window = args.context_window
                    model_prediction_window = args.prediction_window
                    model = RadRecurrent(data_dim=model_data_dim, lstm_dim=model_lstm_dim, lstm_depth=model_lstm_depth, dropout=model_dropout, context_window=model_context_window, prediction_window=model_prediction_window)
                elif args.model_type == 'RadRecurrentWithSDO':
                    model_data_dim_context = 2
                    model_data_dim_prediction = 1
                    model_lstm_dim = 1024
                    model_lstm_depth = args.lstm_depth
                    model_dropout = 0.2
                    model_sdo_dim = 1024
                    model_sdo_channels = 6
                    model_context_window = args.context_window
                    model_prediction_window = args.prediction_window
                    model = RadRecurrentWithSDO(data_dim_context=model_data_dim_context, 
                                                data_dim_predict=model_data_dim_prediction,
                                                lstm_dim=model_lstm_dim, 
                                                lstm_depth=model_lstm_depth, 
                                                dropout=model_dropout, 
                                                sdo_channels=model_sdo_channels, 
                                                sdo_dim=model_sdo_dim, 
                                                context_window=model_context_window, 
                                                prediction_window=model_prediction_window, 
                                                # sdo_only_context=args.sdo_only_context ## This was for ablation but we implemented it differently now.
                                            )
                elif args.model_type == 'RadRecurrentWithSDOCore':
                    model_data_dim_context = 2
                    model_data_dim_prediction = 1
                    model_lstm_dim = 1024
                    model_lstm_depth = args.lstm_depth
                    model_dropout = 0.2 
                    model_context_window = args.context_window
                    model_prediction_window = args.prediction_window
                    model = RadRecurrentWithSDOCore(data_dim_context=model_data_dim_context, 
                                                    data_dim_predict=model_data_dim_prediction, 
                                                    lstm_dim=model_lstm_dim, 
                                                    lstm_depth=model_lstm_depth, 
                                                    dropout=model_dropout, 
                                                    context_window=model_context_window, 
                                                    prediction_window=model_prediction_window)

                else:
                    raise ValueError('Unknown model type: {}'.format(args.model_type))
                
                optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                iteration = 0
                epoch_start = 0
                train_losses = []
                valid_losses = []
                model = model.to(device)


            # We are using Monte Carlo dropout https://arxiv.org/abs/1506.02142 which requires dropout to be used at all times, including test/prediction/inference time
            # Never use model.eval() at all, because it will disable dropout
            model.train()

            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('\nNumber of parameters: {:,}\n'.format(num_params))

            ## Open file for saving epoch-wise losses
            epoch_losses_filepath = '{}/epoch_losses.txt'.format(main_study_dir+"/train/loss", epoch_start+1)
            epoch_losses_file = open(epoch_losses_filepath, 'a')
            ###epoch_losses_file.write('Epoch TrainLoss ValidLoss\n')
            for epoch in range(epoch_start, args.epochs):
                epoch_start = datetime.datetime.now()
                print('\n*** Epoch {:,}/{:,} started {}'.format(epoch+1, args.epochs, epoch_start))
                print('*** Training')
                # model.train()
                with tqdm(total=len(train_loader)) as pbar:
                    for i, batch in enumerate(train_loader):
                        
                        if isinstance(model, RadRecurrentWithSDO):
                            (sdo, goesxrs, biosentinel, _) = batch
                            batch_size = goesxrs.shape[0]

                            sdo = sdo.to(device)
                            goesxrs = goesxrs.to(device)
                            biosentinel = biosentinel.to(device)
                            goesxrs = goesxrs.unsqueeze(-1)
                            biosentinel = biosentinel.unsqueeze(-1)
                            data = torch.cat([goesxrs, biosentinel], dim=2)

                            # context window
                            context_sdo = sdo[:, :model.context_window]
                            context_data = data[:, :model.context_window]

                            # prediction window
                            input = data[:, model.context_window:-1,-1]
                            target = data[:, model.context_window+1:,-1]
                            input=input.unsqueeze(-1)
                            target=target.unsqueeze(-1)

                            model.init(batch_size)
                            optimizer.zero_grad()
                            model.forward_context(context_sdo, context_data)
                            output = model.forward(input)
                        
                        elif isinstance(model, RadRecurrentWithSDOCore):
                            (sdo, goesxrs, biosentinel, _) = batch
                            batch_size = goesxrs.shape[0]

                            sdo = sdo.to(device)
                            goesxrs = goesxrs.to(device)
                            biosentinel = biosentinel.to(device)
                            goesxrs = goesxrs.unsqueeze(-1)
                            biosentinel = biosentinel.unsqueeze(-1)
                            data = torch.cat([goesxrs, biosentinel], dim=2)

                            # context window
                            context_sdo = sdo[:, :model.context_window]
                            context_data = data[:, :model.context_window]

                            # prediction window
                            input = data[:, model.context_window:-1,-1]
                            target = data[:, model.context_window+1:,-1]
                            input=input.unsqueeze(-1)
                            target=target.unsqueeze(-1)

                            model.init(batch_size)
                            optimizer.zero_grad()
                            model.forward_context(context_sdo, context_data)
                            output = model.forward(input)


                        elif isinstance(model, RadRecurrent): ## This is deprecated now, will be removed in the future.
                            (goesxrs, biosentinel, _) = batch
                            batch_size = goesxrs.shape[0]

                            goesxrs = goesxrs.to(device)
                            biosentinel = biosentinel.to(device)
                            goesxrs = goesxrs.unsqueeze(-1)
                            biosentinel = biosentinel.unsqueeze(-1)
                            data = torch.cat([goesxrs, biosentinel], dim=2)
                            
                            input = data[:, :-1]
                            target = data[:, 1:]
                        
                            model.init(batch_size)
                            optimizer.zero_grad()
                            output = model(input)
                        else:
                            raise ValueError('Unknown model type: {}'.format(model))
                        
                        loss = torch.nn.functional.mse_loss(output, target)
                        loss.backward()
                        optimizer.step()

                        ##### Intermediate outputs #####
                        train_loss_file = '{}/epoch-{:03d}-train_loss.txt'.format(main_study_dir+"/train/loss", epoch+1)
                        f = open(train_loss_file, 'a')
                        f.write('%d %.5e\n'%(iteration, float(loss)))
                        f.close()
                        # # Save model
                        # if iteration%50==0:
                        #     model_file_intermediate = '{}/intermediate_epoch-{:03d}-model.pth'.format(main_study_dir+"/train/saved_model", epoch+1)
                        #     save_model(model, optimizer, epoch, iteration, train_losses, valid_losses, model_file_intermediate)
                        ############

                        train_losses.append((iteration, float(loss)))

                        pbar.set_description('Epoch: {:,}/{:,} | Iter: {:,}/{:,} | Loss: {:.4f}'.format(epoch+1, args.epochs, i+1, len(train_loader), float(loss)))
                        pbar.update(1)

                        iteration += 1
                
                print('*** Validation')
                with torch.no_grad():
                    valid_loss = 0.
                    valid_seqs = 0
                    # model.eval()
                    with tqdm(total=len(valid_loader), desc='Validation') as pbar:
                        for batch in valid_loader:
                            
                            if isinstance(model, RadRecurrentWithSDO):
                                (sdo, goesxrs, biosentinel, _) = batch
                                batch_size = goesxrs.shape[0]

                                sdo = sdo.to(device)
                                goesxrs = goesxrs.to(device)
                                biosentinel = biosentinel.to(device)
                                goesxrs = goesxrs.unsqueeze(-1)
                                biosentinel = biosentinel.unsqueeze(-1)
                                data = torch.cat([goesxrs, biosentinel], dim=2)

                                context_sdo = sdo[:, :model.context_window]
                                context_data = data[:, :model.context_window]

                                input = data[:, model.context_window:-1,-1]
                                target = data[:, model.context_window+1:,-1]
                                input=input.unsqueeze(-1)
                                target=target.unsqueeze(-1)

                                model.init(batch_size)
                                model.forward_context(context_sdo, context_data)
                                output = model.forward(input)

                            elif isinstance(model, RadRecurrentWithSDOCore):
                                (sdo, goesxrs, biosentinel, _) = batch
                                batch_size = goesxrs.shape[0]

                                sdo = sdo.to(device)
                                goesxrs = goesxrs.to(device)
                                biosentinel = biosentinel.to(device)
                                goesxrs = goesxrs.unsqueeze(-1)
                                biosentinel = biosentinel.unsqueeze(-1)
                                data = torch.cat([goesxrs, biosentinel], dim=2)

                                # context window
                                context_sdo = sdo[:, :model.context_window]
                                context_data = data[:, :model.context_window]

                                # prediction window
                                input = data[:, model.context_window:-1,-1]
                                target = data[:, model.context_window+1:,-1]
                                input=input.unsqueeze(-1)
                                target=target.unsqueeze(-1)

                                model.init(batch_size)
                                model.forward_context(context_sdo, context_data)
                                output = model.forward(input)

                            elif isinstance(model, RadRecurrent):
                                (goesxrs, biosentinel, _) = batch
                                batch_size = goesxrs.shape[0]

                                goesxrs = goesxrs.to(device)
                                biosentinel = biosentinel.to(device)
                                goesxrs = goesxrs.unsqueeze(-1)
                                biosentinel = biosentinel.unsqueeze(-1)
                                data = torch.cat([goesxrs, biosentinel], dim=2)

                                input = data[:, :-1]
                                target = data[:, 1:]

                                model.init(batch_size)
                                output = model(input)
                            else:
                                raise ValueError('Unknown model type: {}'.format(model))
                            
                            loss = torch.nn.functional.mse_loss(output, target)
                            valid_loss += float(loss)
                            valid_seqs += 1
                            pbar.update(1)

                    if valid_seqs == 0:
                        valid_loss = 0.
                    else:
                        valid_loss /= valid_seqs
                    print('\nEpoch: {:,}/{:,} | Iter: {:,}/{:,} | Loss: {:.4f} | Valid loss: {:.4f}'.format(epoch+1, args.epochs, i+1, len(train_loader), float(loss), valid_loss))
                    valid_losses.append((iteration, valid_loss))
                
                
                ## Append the epoch-wise losses to the file
                mean_train_loss = np.mean([loss for _, loss in train_losses])
                epoch_losses_file.write('%d %.5e %.5e\n'%(epoch, mean_train_loss, valid_loss))
                

                # Save model
                model_file = '{}/epoch-{:03d}-model.pth'.format(main_study_dir+"/saved_model", epoch+1)
                save_model(model, optimizer, epoch, iteration, train_losses, valid_losses, model_file)

                
                # Plot losses
                plot_file = '{}/epoch-{:03d}-loss.pdf'.format(main_study_dir+"/train/plots", epoch+1)
                save_loss_plot(train_losses, valid_losses, plot_file)

                # if args.test_event_id is not None:
                #     for event_id in args.test_event_id:
                #         if event_id not in EventCatalog:
                #             raise ValueError('Event ID not found in events: {}'.format(event_id))
                #         date_start, date_end = EventCatalog[event_id]
                #         print('\nEvent ID: {}'.format(event_id))
                #         date_start = datetime.datetime.fromisoformat(date_start)
                #         date_end = datetime.datetime.fromisoformat(date_end)
                #         file_prefix = 'epoch-{:03d}-test-event-{}-{}-{}'.format(epoch+1, event_id, date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))
                #         title = 'Event: {} '.format(event_id)
                #         plot_ylims = run_test(model, date_start, date_end, file_prefix, title, args)
                #         run_test_video(model, date_start, date_end, file_prefix, title, plot_ylims, args)

                # if args.test_seen_event_id is not None:
                #     for event_id in args.test_seen_event_id:
                #         if event_id not in EventCatalog:
                #             raise ValueError('Event ID not found in events: {}'.format(event_id))
                #         date_start, date_end = EventCatalog[event_id]
                #         print('\nEvent ID: {}'.format(event_id))
                #         date_start = datetime.datetime.fromisoformat(date_start)
                #         date_end = datetime.datetime.fromisoformat(date_end)
                #         file_prefix = 'epoch-{:03d}-test-seen-event-{}-{}-{}'.format(epoch+1, event_id, date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))
                #         title = 'Event: {} '.format(event_id)
                #         plot_ylims = run_test(model, date_start, date_end, file_prefix, title, args)
                #         run_test_video(model, date_start, date_end, file_prefix, title, plot_ylims, args)

                epoch_end = datetime.datetime.now()
                print('\n*** Epoch {:,}/{:,} ended {}, duration {}'.format(epoch+1, args.epochs, epoch_end, epoch_end - epoch_start))

        if args.mode == 'test':
            print('\n*** Testing mode\n')

            model, _, _, _, _, _ = load_model(args.model_file, device)
            model.train() # set to train mode to use MC dropout
            model.to(device)

            tests_to_run = []
            if args.test_event_id is not None:
                print('\nEvent IDs given, will ignore date_start and date_end arguments and use event dates')

                for event_id in args.test_event_id:
                    if event_id not in EventCatalog:
                        raise ValueError('Event ID not found in events: {}'.format(event_id))
                    date_start, date_end = EventCatalog[event_id]
                    print('\nEvent ID: {}'.format(event_id))

                    date_start = datetime.datetime.fromisoformat(date_start)
                    date_end = datetime.datetime.fromisoformat(date_end)
                    file_prefix = 'test-event-{}-{}'.format(date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))
                    title = 'Event: {} '.format(event_id)
                    tests_to_run.append((date_start, date_end, file_prefix, title))

            else:
                print('\nEvent IDs not given, will use date_start and date_end arguments')

                date_start = datetime.datetime.fromisoformat(args.date_start)
                date_end = datetime.datetime.fromisoformat(args.date_end)
                file_prefix = 'test-event-{}-{}'.format(date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))
                title = None
                tests_to_run.append((date_start, date_end, file_prefix, title))


            for date_start, date_end, file_prefix, title in tests_to_run:
                save_test_numpy(model, date_start, date_end, main_study_dir, file_prefix, args)
                #plot_ylims = run_test(model, date_start, date_end, file_prefix, title, args)
                #run_test_video(model, date_start, date_end, file_prefix, title, plot_ylims, args)


        print('\nEnd time: {}'.format(datetime.datetime.now()))
        print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == "__main__":
    main()