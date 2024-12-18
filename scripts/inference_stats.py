import numpy as np
import os 
import argparse
import sys
from pylab import *
import matplotlib.pyplot as plt
from events import EventCatalog as EventCatalogTest
from events_months import EventCatalog
import datetime
import matplotlib.gridspec as gridspec

def main():
    description = 'FDL-X 2024, Radiation Team, preliminary machine learning experiments'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to store results')#/home/emassara/2024-hl-radiation-ml/results
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory with datasets')#/mnt/disks/hl-dosi-datasets/data/
    parser.add_argument('--solar_dataset', type=str, choices=['SDOMLlite', 'SDOCore'], default='SDOMLlite', help='Solar dataset type')
    parser.add_argument('--rad_inst', type=str, choices=['CRaTER-D1D2','BPD'], default='CRaTER-D1D2', help='Radiation instrument')
    parser.add_argument('--sdo_random_data', action='store_true', help='Use fake SDO data (for ablation study)')
    parser.add_argument('--xray_random_data', action='store_true', help='Use fake xray data (for ablation study)')
    parser.add_argument('--rad_random_data', action='store_true', help='Use fake radiation data (for ablation study)')
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
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--lstm_depth', type=int, default=2, help='LSTM depth')
    parser.add_argument('--model_type', type=str, choices=['RadRecurrent', 'RadRecurrentWithSDO','RadRecurrentWithSDOCore'], default='RadRecurrentWithSDO', help='Model type')
    parser.add_argument('--date_start', type=str, default='2022-11-16T11:00:00', help='Start date') #default='2022-11-16T11:00:00' '2017-02-07T00:00:00'
    parser.add_argument('--date_end', type=str, default='2024-05-14T09:15:00', help='End date')     #default='2024-05-14T09:15:00' '2024-05-31T23:59:59'
    parser.add_argument('--test_event_id', nargs='+', default=['crater54', 'crater68', 'crater69'], help='Test event IDs')
    parser.add_argument('--valid_event_id', nargs='+', default=['valid14','valid15'], help='Validation event IDs')
    parser.add_argument('--train_event_id', nargs='+', default=None, help='Test event IDs seen during training')  #'biosentinel07','biosentinel08','biosentinel14','biosentinel15'

    parser.add_argument('--model_file', type=str, help='Model file')
    parser.add_argument('--multiples_prediction_window', type=int, default=1, help='multiples_prediction_window')
    parser.add_argument('--output', type=str, choices=['rad-xray','rad'], required=True, help='type of output')

    args = parser.parse_args()
    if args.solar_dataset == 'SDOMLlite':
        args.sdo_dir = 'sdoml-lite'
    elif args.solar_dataset == 'SDOCore':
        args.sdo_dir = 'sdocore'

    # make sure the target directory exists
    os.makedirs(args.target_dir, exist_ok=True)

    ## Create the ablation study ids
    if args.output=='rad-xray':
        if not args.sdo_random_data and not args.xray_random_data and not args.rad_random_data:
            study_id = "study--solar-rad-xray_to_rad-xray" ## This is the "max" study
        elif not args.sdo_random_data and not args.xray_random_data and args.rad_random_data:
            study_id = "study--solar-xray_to_rad-xray"
        elif not args.sdo_random_data and args.xray_random_data and not args.rad_random_data:
            study_id = "study--solar-rad_to_rad-xray"
        elif args.sdo_random_data and not args.xray_random_data and not args.rad_random_data:
            study_id = "study--rad-xray_to_rad-xray"
    elif args.output=='rad':
        if not args.sdo_random_data and not args.xray_random_data and not args.rad_random_data:
            study_id = "study--solar-rad-xray_to_rad" ## This is the "max" study
        elif not args.sdo_random_data and not args.xray_random_data and args.rad_random_data:
            study_id = "study--solar-xray_to_rad"
        elif not args.sdo_random_data and args.xray_random_data and not args.rad_random_data:
            study_id = "study--solar-rad_to_rad"
        elif args.sdo_random_data and not args.xray_random_data and not args.rad_random_data:
            study_id = "study--rad-xray_to_rad"
    
    ## Study directory with the folder structure as decided: /home/username/2024-hl-radiation-ml/results/solar-inst[SDOCORE]/rad-inst[CRaTER]/study–solar-rad-xray_to_rad
    main_study_dir = args.target_dir+f"/solar-dataset[{args.solar_dataset}]/rad-inst[{args.rad_inst}]/{study_id}/{args.date_start}-{args.date_end}"
    
    dir_model_epoch = args.model_file.split("/")[-1].split(".")[0]
    dir_test_pred = main_study_dir+'/test/saved_predictions/'+dir_model_epoch
    dir_test_plot = main_study_dir+'/'+'test/plots/'+dir_model_epoch

    ## Read the predicitons of the form [# now_times, # dropuot_realizations, # prediction_timestamps]
    list_radp_filename = [filename for filename in os.listdir(dir_test_pred) if filename.startswith("test-rad_pred") and filename.endswith("%dwprediction.npy"%args.multiples_prediction_window)]                            
    list_radp_filename.sort()
    data_rad = np.empty((0,args.num_samples,args.prediction_window*args.multiples_prediction_window+1))
    print("%dwprediction.npy"%args.multiples_prediction_window)
    for filename in list_radp_filename:
        data_i = np.load(dir_test_pred+'/'+filename,allow_pickle=True)
        data_rad = np.append(data_rad,data_i,axis=0)
    
    ## Read the ground truth
    list_radt_filename = [filename for filename in os.listdir(dir_test_pred) if filename.startswith("test-rad_truth") and filename.endswith("%dwprediction.npy"%args.multiples_prediction_window)]                            
    list_radt_filename.sort()
    truth_rad = np.empty((0,args.prediction_window*args.multiples_prediction_window+1))
    print("%dwprediction.npy"%args.multiples_prediction_window)
    for filename in list_radt_filename:
        data_i = np.load(dir_test_pred+'/'+filename,allow_pickle=True)
        truth_rad = np.append(truth_rad,data_i,axis=0)

    ## Read the dates
    list_date_filename = [filename for filename in os.listdir(dir_test_pred) if filename.startswith("test-date") and filename.endswith("%dwprediction.npy"%args.multiples_prediction_window)]                            
    list_date_filename.sort()
    date = np.empty((0,args.num_samples,args.prediction_window*args.multiples_prediction_window+1))
    print("%dwprediction.npy"%args.multiples_prediction_window)
    for filename in list_date_filename:
        data_i = np.load(dir_test_pred+'/'+filename,allow_pickle=True)
        date = np.append(date,data_i,axis=0)

    ## get the locatios of nowtimes inside events
    test_events = np.empty(0)
    if args.test_event_id is not None:
        #print('\nEvent IDs given, will ignore date_start and date_end arguments and use event dates')

        for event_id in args.test_event_id:
            if event_id in EventCatalogTest:
                date_start, date_end, _ = EventCatalogTest[event_id]
                print('\nEvent ID: {}'.format(event_id))
            else:
                print('Event ID not found in events: {}'.format(event_id))
                continue
            date_start = datetime.datetime.fromisoformat(date_start)
            date_end = datetime.datetime.fromisoformat(date_end)
            sel_events = np.where((date[:,0,1]>date_start)&(date[:,0,1]<date_end))[0]
            test_events = np.append(test_events,sel_events)
            print(test_events)
        test_events=np.array(test_events,dtype=np.int16)

    ### compute variance of the model ###
    var_rad_nowtime = np.var(data_rad,axis=1)
    print('var',var_rad_nowtime.shape)  #should be N,121
    var_rad = np.mean(var_rad_nowtime,axis=0) #should be of size 121

    ### compute the MSE ###
    mean_rad_nowtime = np.mean(data_rad,axis=1)
    mse_rad = np.nanmean((mean_rad_nowtime-truth_rad)**2,axis=0)
    # events only
    mean_rad_nowtime_events = np.mean(data_rad[test_events],axis=1)
    mse_rad_events = np.nanmean((mean_rad_nowtime_events-truth_rad[test_events])**2,axis=0)
    ### compute var(mean) ###
    mean_ground_truth = np.nanmean(truth_rad) # consider multiple time each timestamp as inside each datapoint
    var_rad_mean = np.mean((mean_rad_nowtime-mean_ground_truth)**2,axis=0)
    r_squared = 1 - mse_rad/var_rad_mean
    # events only
    #mean_ground_truth_events = np.nanmean(truth_rad[test_events]) # consider multiple time each timestamp as inside each datapoint in event time
    var_rad_mean_events = np.mean((mean_rad_nowtime_events-mean_ground_truth)**2,axis=0)
    r_squared_events = 1 - mse_rad_events/var_rad_mean_events

    print('ground truth mean of all test set and during events only:')
    #print(mean_ground_truth,mean_ground_truth_events)


    ### FIGURE RMSE + STD MODEL ###
    fig = figure(figsize=(7,4))
    gs = gridspec.GridSpec(1,1)
    axs = []
    axs.append(fig.add_subplot(gs[0]))
    x_list = np.arange(len(var_rad))*15/60
    axs[0].set_xlabel(r"Hours from present time")
    axs[0].set_ylabel(r"Error [$\mu$Gr/hr]")
    axs[0].plot(x_list[x_list<=10],mse_rad[x_list<=10]**0.5,label="RMSE",color='C0')
    axs[0].plot(x_list[x_list<=10],mse_rad_events[x_list<=10]**0.5,label="RMSE during events",color='C1')
    axs[0].plot(x_list[x_list>=10],mse_rad[x_list>=10]**0.5,alpha=0.5,color='C0')
    axs[0].plot(x_list[x_list>=10],mse_rad_events[x_list>=10]**0.5,alpha=0.5,color='C1')
    axs[0].fill_between(x_list,var_rad**0.5,label="Model uncertainty - 1 std",color='k',alpha=0.4)
    axs[0].fill_between(x_list,2*var_rad**0.5,label="Model uncertainty - 2 std",color='k',alpha=0.2)
    #axs[0].fill_between(x_list[x_list>=10],np.ones(len(x_list[x_list>=10]))+700,color='grey',alpha=0.1)
    axs[0].axvspan(10, 30, alpha=0.1, color='grey')
    axs[0].axvline(10,color='k',lw=0.6,ls='--')
    axs[0].legend(loc = "upper left",
                prop={'size':10})
    plt.savefig(dir_test_plot+'/fig_test_error-time_from_nowtime_%dwprediction.pdf'%args.multiples_prediction_window)


    ### FIGURE Rsquared as function of hours from nowtime ###
    fig = figure(figsize=(7,4))
    gs = gridspec.GridSpec(1,1)
    axs = []
    axs.append(fig.add_subplot(gs[0]))
    x_list = np.arange(len(var_rad))*15/60
    axs[0].set_xlabel(r"Hours from present time")
    axs[0].set_ylabel(r"$R^2$")
    axs[0].set_ylim(-1,1)
    # axs[0].plot(x_list,var_rad_mean_events)
    # axs[0].plot(x_list,var_rad_mean)
    axs[0].plot(x_list[x_list<=10],r_squared[x_list<=10],label="All times in test set",color='C0')
    axs[0].plot(x_list[x_list<=10],r_squared_events[x_list<=10],label="Event times in test set",color='C1')
    #axs[0].plot(x_list[x_list>=10],r_squared[x_list>=10],alpha=0.5,color='C0')
    #axs[0].plot(x_list[x_list>=10],r_squared_events[x_list>=10],alpha=0.5,color='C1')
    #axs[0].axvspan(10, 30, alpha=0.1, color='grey')
    #axs[0].axvline(10,color='k',lw=0.6,ls='--')
    axs[0].legend(loc = "upper right",
                prop={'size':10})
    plt.savefig(dir_test_plot+'/fig_test_rsquared-time_from_nowtime_%dwprediction.pdf'%args.multiples_prediction_window)
    

if __name__ == "__main__":
    main()
  