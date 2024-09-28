import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
import os
import datetime
from tqdm import tqdm
import pandas as pd
from io import BytesIO
import tarfile
import pickle
from functools import lru_cache
import duckdb
import math


class SDOMLlite(Dataset):
    def __init__(self, data_dir, channels=['hmi_m', 'aia_0131', 'aia_0171', 'aia_0193', 'aia_0211', 'aia_1600'], date_start=None, date_end=None, date_exclusions=None, random_data=False):
        self.data_dir = data_dir
        self.channels = channels
        print('\nSDOML-lite')

        self.random_data = random_data
        if random_data:
            self.random_data_shape = torch.Size([len(self.channels), 512, 512])
            print('Random data: True')

        print('Directory  : {}'.format(self.data_dir))

        self.data = WebDataset(data_dir)

        self.date_start, self.date_end = self.find_date_range()
        if date_start is not None:
            if isinstance(date_start, str):
                date_start = datetime.datetime.fromisoformat(date_start)
            
            if (date_start >= self.date_start) and (date_start < self.date_end):
                self.date_start = date_start
            else:
                print('Start date out of range, using default')
        if date_end is not None:
            if isinstance(date_end, str):
                date_end = datetime.datetime.fromisoformat(date_end)
            if (date_end > self.date_start) and (date_end <= self.date_end):
                self.date_end = date_end
            else:
                print('End date out of range, using default')
        self.delta_minutes = 15
        total_minutes = int((self.date_end - self.date_start).total_seconds() / 60)
        total_steps = total_minutes // self.delta_minutes
        print('Start date : {}'.format(self.date_start))
        print('End date   : {}'.format(self.date_end))
        print('Delta      : {} minutes'.format(self.delta_minutes))
        print('Channels   : {}'.format(', '.join(self.channels)))

        self.date_exclusions = date_exclusions
        if self.date_exclusions is not None:
            print('Date exclusions:')
            date_exclusions_postfix = '_exclusions'
            for exclusion_date_start, exclusion_date_end in self.date_exclusions:
                print('  {} - {}'.format(exclusion_date_start, exclusion_date_end))
                date_exclusions_postfix += '__{}_{}'.format(exclusion_date_start.isoformat(), exclusion_date_end.isoformat())
        else:
            date_exclusions_postfix = ''

        if self.random_data:
            self.dates = []
            for i in range(total_steps):
                date = self.date_start + datetime.timedelta(minutes=self.delta_minutes*i)
                exists = True
                if self.date_exclusions is not None:
                    for exclusion_date_start, exclusion_date_end in self.date_exclusions:
                        if (date >= exclusion_date_start) and (date < exclusion_date_end):
                            exists = False
                            break
                if exists:
                    self.dates.append(date)
        else:
            self.dates = []
            dates_cache = 'dates_index_{}_{}_{}{}'.format('_'.join(self.channels), self.date_start.isoformat(), self.date_end.isoformat(), date_exclusions_postfix)
            dates_cache = os.path.join('./../results/', dates_cache) #changed #self.data_dir, dates_cache)
            if os.path.exists(dates_cache):
                print('Loading dates from cache: {}'.format(dates_cache))
                with open(dates_cache, 'rb') as f:
                    self.dates = pickle.load(f)
            else:        
                for i in tqdm(range(total_steps), desc='Checking complete channels'):
                    date = self.date_start + datetime.timedelta(minutes=self.delta_minutes*i)
                    exists = True
                    prefix = self.date_to_prefix(date)
                    data = self.data.index.get(prefix)
                    if data is None:
                        exists = False
                    else:
                        for channel in self.channels:
                            postfix = channel+'.npy'
                            if postfix not in data:
                                exists = False
                                break
                    if self.date_exclusions is not None:
                        for exclusion_date_start, exclusion_date_end in self.date_exclusions:
                            if (date >= exclusion_date_start) and (date < exclusion_date_end):
                                exists = False
                                break
                    if exists:
                        self.dates.append(date)
                print('Saving dates to cache: {}'.format(dates_cache))
                with open(dates_cache, 'wb') as f:
                    pickle.dump(self.dates, f)
            
        if len(self.dates) == 0:
            raise RuntimeError('No frames found with given list of channels')

        self.dates_set = set(self.dates)
        self.name = 'SDOML-lite'

        print('Frames total    : {:,}'.format(total_steps))
        print('Frames available: {:,}'.format(len(self.dates)))
        print('Frames dropped  : {:,}'.format(total_steps - len(self.dates)))


    @lru_cache(maxsize=100000)
    def prefix_to_date(self, prefix):
        return datetime.datetime.strptime(prefix, '%Y/%m/%d/%H%M')
    
    @lru_cache(maxsize=100000)
    def date_to_prefix(self, date):
        return date.strftime('%Y/%m/%d/%H%M')

    def find_date_range(self):
        if self.random_data:
            date_start = datetime.datetime.fromisoformat('2010-05-13T00:00:00')
            date_end = datetime.datetime.fromisoformat('2024-07-27T00:00:00')
        else:
            prefix_start = self.data.prefixes[0]
            prefix_end = self.data.prefixes[-1]
            date_start = self.prefix_to_date(prefix_start)
            date_end = self.prefix_to_date(prefix_end)
        return date_start, date_end
    
    def __repr__(self):
        return 'SDOML-lite ({} - {})'.format(self.date_start, self.date_end)


    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            date = self.dates[index]
        elif isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        else:
            raise ValueError('Expecting index to be int, datetime.datetime, or str (in the format of 2022-11-01T00:01:00)')
        data = self.get_data(date)    
        return data, date.isoformat()
    
    def get_data(self, date):
        # if date < self.date_start or date > self.date_end:
        #     raise ValueError('Date ({}) out of range for SDOML-lite ({} - {})'.format(date, self.date_start, self.date_end))

        if date not in self.dates_set:
            print('Date not found in SDOML-lite : {}'.format(date))
            return None
        
        if self.date_exclusions is not None:
            for exclusion_date_start, exclusion_date_end in self.date_exclusions:
                if (date >= exclusion_date_start) and (date < exclusion_date_end):
                    raise RuntimeError('Should not happen')

        if self.random_data:
            # print('SDOML-lite returning random data for date: {}'.format(date))
            channels = torch.randn(self.random_data_shape)
        else:
            prefix = self.date_to_prefix(date)
            data = self.data[prefix]
            channels = []
            for channel in self.channels:
                file = channel+'.npy'
                channel_data = data[file]
                channels.append(channel_data)
            channels = np.stack(channels)
            channels = torch.from_numpy(channels)
        return channels

    # For plotting purposes only, produced values cannot be used for any scientific-quality data analysis
    def unnormalize(self, data, channel):
        sqrt_aia_cutoff = {}
        sqrt_aia_cutoff['aia_0131'] = math.sqrt(2652.1470)
        sqrt_aia_cutoff['aia_0171'] = math.sqrt(22816.1035)
        sqrt_aia_cutoff['aia_0193'] = math.sqrt(23919.7168)
        sqrt_aia_cutoff['aia_0211'] = math.sqrt(13458.3203)
        sqrt_aia_cutoff['aia_1600'] = math.sqrt(3399.5896)
        if channel == 'hmi_m':
            mask = data > 0.05
            data = 2 * (data - 0.5)
            data = data * 1500 
            data = data * mask
        else:
            c = sqrt_aia_cutoff[channel]
            data = data * c
            data = data ** 2.
        return data


class PandasDataset(Dataset):
    def __init__(self, name, data_frame, column, delta_minutes, date_start=None, date_end=None, normalize=True, rewind_minutes=15, date_exclusions=None):
        self.name = name
        self.data = data_frame
        self.column = column
        self.delta_minutes = delta_minutes

        print('Delta minutes        : {:,}'.format(self.delta_minutes))
        self.normalize = normalize
        print('Normalize            : {}'.format(self.normalize))
        self.rewind_minutes = rewind_minutes
        print('Rewind minutes       : {:,}'.format(self.rewind_minutes))

        self.data[column] = self.data[column].astype(np.float32)

        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data = self.data.dropna()

        # Get dates available
        self.dates = [date.to_pydatetime() for date in self.data['datetime']]
        self.dates_set = set(self.dates)
        self.date_start = self.dates[0]
        self.date_end = self.dates[-1]

        # Adjust dates available
        if date_start is not None:
            if isinstance(date_start, str):
                date_start = datetime.datetime.fromisoformat(date_start)
            if (date_start >= self.date_start) and (date_start < self.date_end):
                self.date_start = date_start
            else:
                print('Start date out of range, using default')
        if date_end is not None:
            if isinstance(date_end, str):
                date_end = datetime.datetime.fromisoformat(date_end)
            if (date_end > self.date_start) and (date_end <= self.date_end):
                self.date_end = date_end
            else:
                print('End date out of range, using default')        

        if not 'CRaTER' in self.name: # Very bad hack, need to fix this for CRaTER
            # if the date of the first row of data after self.date_start does not end in minutes :00, :15, :30, or :45, move forward to the next minute that does
            time_out = 1000
            while True:
                first_row = self.data[self.data['datetime'] >= self.date_start].iloc[0]
                first_row_date = first_row['datetime']
                if first_row_date.minute % 15 != 0:
                    print('Adjust startdate(old): {}'.format(first_row_date))
                    first_row_date = first_row_date + datetime.timedelta(minutes=15 - (first_row_date.minute % 15))
                    print('Adjust startdate(new): {}'.format(first_row_date))
                    self.date_start = first_row_date
                    time_out -= 1
                    if time_out == 0:
                        raise RuntimeError('Time out in adjusting start date for {}'.format(self.name))
                else:
                    break

        # Filter out dates outside the range
        self.data = self.data[(self.data['datetime'] >=self.date_start) & (self.data['datetime'] <=self.date_end)]

        # Filter out dates within date_exclusions
        self.date_exclusions = date_exclusions
        if self.date_exclusions is not None:
            print('Date exclusions:')
            for exclusion_date_start, exclusion_date_end in self.date_exclusions:
                print('  {} - {}'.format(exclusion_date_start, exclusion_date_end))
                self.data = self.data[~self.data['datetime'].between(exclusion_date_start, exclusion_date_end)]

        # Get dates available (redo to make sure things match up)
        self.dates = [date.to_pydatetime() for date in self.data['datetime']]
        self.dates_set = set(self.dates)
        self.date_start = self.dates[0]
        self.date_end = self.dates[-1]
        print('Start date           : {}'.format(self.date_start))
        print('End date             : {}'.format(self.date_end))

        print('Rows after processing: {:,}'.format(len(self.data)))
        # print('Memory usage         : {:,} bytes'.format(self.data.memory_usage(deep=True).sum()))



    def normalize_data(self, data):
        raise NotImplementedError('normalize_data not implemented')
    
    def unnormalize_data(self, data):
        raise NotImplementedError('unnormalize_data not implemented')
    
    def __repr__(self):
        return '{} ({} - {})'.format(self.name, self.date_start, self.date_end)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        elif isinstance(index, int):
            date = self.data.iloc[index]['datetime']
        else:
            raise ValueError('Expecting index to be int, datetime.datetime, or str (in the format of 2022-11-01T00:01:00)')
        data = self.get_data(date)
        return data, date.isoformat()            

    def get_data(self, date):
        # if date < self.date_start or date > self.date_end:
        #     raise ValueError('Date ({}) out of range for RadLab ({}; {} - {})'.format(date, self.instrument, self.date_start, self.date_end))

        if date not in self.dates_set:
            print('{} date not found: {}'.format(self.name, date))
            # find the most recent date available before date, in pandas dataframe self.data
            dates_available = self.data[self.data['datetime'] < date]
            if len(dates_available) > 0:
                date_previous = dates_available.iloc[-1]['datetime']
                if date - date_previous < datetime.timedelta(minutes=self.rewind_minutes):
                    print('{} rewinding to  : {}'.format(self.name, date_previous))
                    data = self.data[self.data['datetime'] == date_previous][self.column]
                else:
                    return None
            else:
                return None
        else:
            data = self.data[self.data['datetime'] == date][self.column]
            if len(data) == 0:
                raise RuntimeError('Should not happen')
        data = torch.tensor(data.values[0])
        if self.normalize:
            data = self.normalize_data(data)

        return data

    def get_series(self, date_start, date_end, delta_minutes=None, omit_missing=True):
        if delta_minutes is None:
            delta_minutes = self.delta_minutes
        dates = []
        values = []
        date = date_start
        while date <= date_end:
            value = self.get_data(date)
            value_available = True
            if value is None:
                if omit_missing:
                    value_available = False
                else:
                    value = torch.tensor(float('nan'))
            if value_available:
                dates.append(date)
                values.append(value)
            date += datetime.timedelta(minutes=delta_minutes)
        if len(dates) == 0:
            # raise ValueError('{} no data found between {} and {}'.format(self.name, self.instrument, date_start, date_end))
            return None, None
        values = torch.stack(values).flatten()
        return dates, values


class UnionDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

        print('\nConcatenated datasets')
        for dataset in self.datasets:
            print('Dataset : {}'.format(dataset))

        # check that there is no overlap in the .dates_set of each dataset
        self.dates_set = set()
        self.date_start = datetime.datetime(9999, 12, 31, 23, 59, 59)
        self.date_end = datetime.datetime(1, 1, 1, 0, 0, 0)
        for dataset in self.datasets:
            for date in dataset.dates_set:
                if date < self.date_start:
                    self.date_start = date
                if date > self.date_end:
                    self.date_end = date
                # if date in self.dates_set:
                #     raise ValueError('Overlap in dates_set between datasets')
                self.dates_set.add(date)

    def __len__(self):
        # return sum([len(dataset) for dataset in self.datasets])
        return len(self.dates_set)

    def __getitem__(self, index):
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        else:
            raise ValueError('Expecting index to be datetime.datetime or str (in the format of 2022-11-01T00:01:00)')
        for dataset in self.datasets:
            value, date = dataset[index]
            if value is not None:
                return value, date
        return None, None


class GOESXRS(PandasDataset):
    def __init__(self, file_name, date_start=None, date_end=None, normalize=True, rewind_minutes=15, date_exclusions=None):
        print('\nGOES X-ray Sensor (XRS)')
        print('File                 : {}'.format(file_name))
        delta_minutes = 1

        data = pd.read_csv(file_name)
        data['datetime'] = pd.to_datetime(data['datetime'])
        # data = data.sort_values(by='datetime')
        print('Rows                 : {:,}'.format(len(data)))

        data = data[data['xrsb2_flux'] > 3e-8]

        super().__init__('GOES X-ray Sensor (XRS)', data, 'xrsb2_flux', delta_minutes, date_start, date_end, normalize, rewind_minutes, date_exclusions)

    def normalize_data(self, data):
        data = torch.log(data + 1e-8)
        mean_log_data = -15.498870138883078
        std_log_data = 1.73287805537894
        data = data - mean_log_data
        data = data / std_log_data
        return data
    
    def unnormalize_data(self, data):
        mean_log_data = -15.498870138883078
        std_log_data = 1.73287805537894
        data = data * std_log_data
        data = data + mean_log_data
        data = torch.exp(data) - 1e-8
        return data


# Data units: particles / (cm^2 . s . sr)
class GOESSGPS(PandasDataset):
    def __init__(self, file_name, date_start=None, date_end=None, normalize=True, rewind_minutes=15, date_exclusions=None, column='>10MeV'):
        print('\nGOES Solar and Galactic Proton Sensors (SGPS) ({})'.format(column))
        print('File                 : {}'.format(file_name))
        delta_minutes = 1

        if column != '>10MeV' and column != '>100MeV':
            raise ValueError('Unsupported column: {}'.format(column))
        self.column = column

        data = pd.read_csv(file_name)
        data['datetime'] = pd.to_datetime(data['datetime'])
        # data = data.sort_values(by='datetime')
        print('Rows                 : {:,}'.format(len(data)))

        data = data[data[column] > 0.]

        super().__init__('GOES Solar and Galactic Proton Sensors (SGPS)', data, column, delta_minutes, date_start, date_end, normalize, rewind_minutes, date_exclusions)

    def normalize_data(self, data):
        if self.column == '>10MeV':
            data = torch.log(data + 1e-8)
            mean_log_data = -10.273473739624023
            std_log_data = 1.8938413858413696
            data = data - mean_log_data
            data = data / std_log_data
            return data
        elif self.column == '>100MeV':
            data = torch.log(data + 1e-8)
            mean_log_data = -13.114526748657227
            std_log_data = 0.6752610802650452
            data = data - mean_log_data
            data = data / std_log_data
            return data
        else:
            raise ValueError('Unsupported column: {}'.format(self.column))
    
    def unnormalize_data(self, data):
        if self.column == '>10MeV':
            mean_log_data = -10.273473739624023
            std_log_data = 1.8938413858413696
            data = data * std_log_data
            data = data + mean_log_data
            data = torch.exp(data) - 1e-8
            return data
        elif self.column == '>100MeV':
            mean_log_data = -13.114526748657227
            std_log_data = 0.6752610802650452
            data = data * std_log_data
            data = data + mean_log_data
            data = torch.exp(data) - 1e-8
            return data
        else:
            raise ValueError('Unsupported column: {}'.format(self.column))


def cube_root(x):
    return torch.sign(x) * torch.pow(torch.abs(x), 1/3.)


class RSTNRadio(PandasDataset):
    def __init__(self, file_name, date_start=None, date_end=None, normalize=True, rewind_minutes=15, date_exclusions=None):
        print('\nRadio Solar Telescope Network (RSTN) Solar Radio Burst')
        print('File                 : {}'.format(file_name))
        delta_minutes = 1

        data = pd.read_csv(file_name)
        data['datetime'] = pd.to_datetime(data['datetime'])
        # data = data.sort_values(by='datetime')
        print('Rows                 : {:,}'.format(len(data)))

        column = '245MHz'
        # Remove outliers based on quantiles, takes care of a giant outlier at 2022-07-22 23:14:00, where 245MHz is 22752400.733333
        q_low = data[column].quantile(0.001)
        q_hi  = data[column].quantile(0.999)
        data = data[(data[column] < q_hi) & (data[column] > q_low)]

        super().__init__('Radio Solar Telescope Network (RSTN) Solar Radio Burst', data, column, delta_minutes, date_start, date_end, normalize, rewind_minutes, date_exclusions)

    def normalize_data(self, data):
        data = cube_root(data)
        mean_cuberoot_data = 2.264420986175537
        std_cuberoot_data = 0.9795352816581726
        data = data - mean_cuberoot_data
        data = data / std_cuberoot_data
        return data
    
    def unnormalize_data(self, data):
        data = data * data * data
        mean_cuberoot_data = 2.264420986175537
        std_cuberoot_data = 0.9795352816581726
        data = data * std_cuberoot_data
        data = data + mean_cuberoot_data
        return data


# BioSentinel: 2022-11-16T11:00:00 - 2024-05-14T09:15:00
class RadLab(PandasDataset):
    def __init__(self, file_name, instrument='BPD', date_start=None, date_end=None, normalize=True, rewind_minutes=15, date_exclusions=None):
        self.instrument = instrument
        
        name = 'RadLab ({})'.format(self.instrument)
        print('\n{}'.format(name))
        print('File                 : {}'.format(file_name))
        dm = {}
        dm['Lidal'] = 5
        dm['MSL-RAD-Surface'] = 17
        dm['ALTEA-Survey'] = 1
        dm['CRaTER-D1D2'] = 60
        dm['BPD'] = 1
        if self.instrument in dm:
            delta_minutes = dm[self.instrument]
        else:
            raise RuntimeError('Unsupported instrument: {}'.format(instrument))
        
        # table names: app_metadata, coordinates, instruments, readings, trajectories
        # readings table
        # columns and types: timestamp (double), instrument_id (varchar), direction (varchar), absorbed_dose_rate (double), dose_equivalent_rate (double),flux (double)
        # distinct values in instrument_id: ALTEA-Survey, MSL-RAD-Surface, DosTel2, RAD-NOD3A5-pre, Liulin-MO-AB-Cruise, Liulin-MO-CD-Circular, TEPC-SMP330-pre, REM-LAB1O3-pre, REM-COL1A2-pre, RAD-NOD2P3-pre, Liulin-MO-AB-Circular, BPD, REM-NOD1SSC22-pre, TEPC-SMP327-pre, Liulin-5-D2, REM-CUPSSC17-pre, RAD-JPM1D5-pre, RAD-COL1A2-pre, Liulin-MO-CD-Elliptic, Liulin-MO-AB-Elliptic, IV-TEPC-NOD2DCQ-pre, CRaTER-D1D2, RAD-LAB1O3-pre, LND, DosTel1, REM-JPM1FD4-pre, IV-TEPC-NOD3FD3-pre, IV-TEPC-SMP328-pre, IV-TEPC-COL1A2-pre, Liulin-5-D1, Liulin-5-D3, REM-Lid, REM-NOD3SSC24-pre, REM-LABSSC8-pre, Liulin-MO-CD-Cruise, IV-TEPC-NOD2PD3-pre, Lidal, IV-TEPC-NOD2PCQ-pr
        con = duckdb.connect(file_name, read_only=True)
        instruments_available = list(con.execute('SELECT DISTINCT instrument_id FROM readings').fetchdf().to_numpy().reshape(-1))
        print('Instruments available: {}'.format(', '.join(instruments_available)))
        print('Instrument selected  : {}'.format(self.instrument))
        
        data = con.execute('SELECT timestamp, instrument_id, absorbed_dose_rate FROM readings where instrument_id=\'{}\''.format(instrument)).fetchdf()
        data = data.sort_values(by='timestamp')
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='s', origin='unix', utc=True).dt.tz_localize(None)
        data = data.drop(columns=['timestamp'])

        print('Rows                 : {:,}'.format(len(data)))

        if self.instrument == 'BPD':
            # remove all rows with 0 absorbed_dose_rate
            data = data[data['absorbed_dose_rate'] > 0]
        elif self.instrument == 'CRaTER-D1D2':
            data = data
        elif self.instrument == 'MSL-RAD-Surface':
            data = data
        else:
            raise RuntimeError('Unsupported instrument: {}'.format(self.instrument))
        
        super().__init__(name, data, 'absorbed_dose_rate', delta_minutes, date_start, date_end, normalize, rewind_minutes, date_exclusions)

    def normalize_data(self, data):
        if self.instrument == 'BPD':
            data = torch.log(data + 1e-8)
            mean_log_data = -1.8559070825576782
            std_log_data = 0.739709734916687
            data = data - mean_log_data
            data = data / std_log_data
            return data
        elif self.instrument == 'CRaTER-D1D2':
            mean_log_data = 2.4395177364349365
            std_log_data = 0.752591073513031
            data = torch.log(data + 1e-8)
            data = data - mean_log_data
            data = data / std_log_data
            return data
        elif self.instrument == 'MSL-RAD-Surface':
            data = torch.log(data + 1e-8)
            mean_log_data = 2.0309619903564453
            std_log_data = 0.217209592461586
            data = data - mean_log_data
            data = data / std_log_data
            return data
        else:
            raise RuntimeError('Unsupported instrument: {}'.format(self.instrument))
    
    def unnormalize_data(self, data):
        if self.instrument == 'BPD':
            mean_log_data = -1.8559070825576782
            std_log_data = 0.739709734916687
            data = data * std_log_data
            data = data + mean_log_data
            data = torch.exp(data) - 1e-8
            return data
        elif self.instrument == 'CRaTER-D1D2':
            mean_log_data = 2.4395177364349365
            std_log_data = 0.752591073513031
            data = data * std_log_data
            data = data + mean_log_data
            data = torch.exp(data) - 1e-8
            return data
        elif self.instrument == 'MSL-RAD-Surface':
            mean_log_data = 2.0309619903564453
            std_log_data = 0.217209592461586
            data = data * std_log_data
            data = data + mean_log_data
            data = torch.exp(data) - 1e-8
            return data
        else:
            raise RuntimeError('Unsupported instrument: {}'.format(self.instrument))


class Sequences(Dataset):
    def __init__(self, datasets, delta_minutes=1, sequence_length=10):
        super().__init__()
        self.datasets = datasets
        self.delta_minutes = delta_minutes
        self.sequence_length = sequence_length

        self.date_start = max([dataset.date_start for dataset in self.datasets])
        self.date_end = min([dataset.date_end for dataset in self.datasets])
        if self.date_start > self.date_end:
            raise ValueError('No overlapping date range between datasets')

        print('\nSequences')
        print('Start date              : {}'.format(self.date_start))
        print('End date                : {}'.format(self.date_end))
        print('Delta                   : {} minutes'.format(self.delta_minutes))
        print('Sequence length         : {}'.format(self.sequence_length))
        print('Sequence duration       : {} minutes'.format(self.delta_minutes*self.sequence_length))

        self.sequences = self.find_sequences()
        if len(self.sequences) == 0:
            print('**** No sequences found ****')
        print('Number of sequences     : {:,}'.format(len(self.sequences)))
        if len(self.sequences) > 0:
            print('First sequence          : {}'.format([date.isoformat() for date in self.sequences[0]]))
            print('Last sequence           : {}'.format([date.isoformat() for date in self.sequences[-1]]))

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        # print('constructing sequence')
        sequence = self.sequences[index]
        sequence_data = []
        for dataset in self.datasets:
            data = []
            for i, date in enumerate(sequence):
                if i == 0:
                    # All data is available at the first step in sequence (by construction of sequences by find_sequence)
                    d, _ = dataset[date]
                    data.append(d)
                else:
                    if date in dataset.dates_set:
                        d, _ = dataset[date]
                        data.append(d)
                    else:
                        data.append(data[i-1])
            data = torch.stack(data)
            sequence_data.append(data)
        sequence_data.append([date.isoformat() for date in sequence])
        # print('done constructing sequence')
        return tuple(sequence_data)


    def find_sequences(self):
        sequences = []
        sequence_start = self.date_start
        while sequence_start <= self.date_end - datetime.timedelta(minutes=(self.sequence_length-1)*self.delta_minutes):
            # New sequence
            sequence = []
            sequence_available = True
            for i in range(self.sequence_length):
                date = sequence_start + datetime.timedelta(minutes=i*self.delta_minutes)
                if i == 0: #nowtime must have all measurements available
                    for dataset in self.datasets:
                        if date not in dataset.dates_set:
                            sequence_available = False
                            break
                if not sequence_available:
                    break
                sequence.append(date)
            if sequence_available:
                sequences.append(sequence)
            # Move to next sequence
            sequence_start += datetime.timedelta(minutes=self.delta_minutes)
        return sequences


class TarRandomAccess():
    def __init__(self, data_dir):
        tar_files = sorted(glob(os.path.join(data_dir, '*.tar')))
        if len(tar_files) == 0:
            raise ValueError('No tar files found in data directory: {}'.format(data_dir))
        self.index = {}
        index_cache = os.path.join('./../results', 'tar_files_index') #changed index_cache = os.path.join(data_dir, 'tar_files_index')
        if os.path.exists(index_cache):
            print('Loading tar files index from cache: {}'.format(index_cache))
            with open(index_cache, 'rb') as file:
                self.index = pickle.load(file)
        else:
            for tar_file in tqdm(tar_files, desc='Indexing tar files'):
                with tarfile.open(tar_file) as tar:
                    for info in tar.getmembers():
                        self.index[info.name] = (tar.name, info)
            print('Saving tar files index to cache: {}'.format(index_cache))
            with open(index_cache, 'wb') as file:
                pickle.dump(self.index, file)
        self.file_names = list(self.index.keys())

    def __getitem__(self, file_name):
        d = self.index.get(file_name)
        if d is None:
            return None
        tar_file, tar_member = d
        with tarfile.open(tar_file) as tar:
            data = BytesIO(tar.extractfile(tar_member).read())
        return data


class WebDataset():
    def __init__(self, data_dir, decode_func=None):
        self.tars = TarRandomAccess(data_dir)
        if decode_func is None:
            self.decode_func = self.decode
        else:
            self.decode_func = decode_func
        
        self.index = {}
        self.prefixes = []
        for file_name in self.tars.file_names:
            p = file_name.split('.', 1)
            if len(p) == 2:
                prefix, postfix = p
                if prefix not in self.index:
                    self.index[prefix] = []
                    self.prefixes.append(prefix)
                self.index[prefix].append(postfix)

    def decode(self, data, file_name):
        if file_name.endswith('.npy'):
            data = np.load(data)
        else:
            raise ValueError('Unknown data type for file: {}'.format(file_name))    
        return data
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        if isinstance(index, str):
            prefix = index
        elif isinstance(index, int):
            prefix = self.prefixes[index]
        else:
            raise ValueError('Expecting index to be int or str')
        sample = self.index.get(prefix)
        if sample is None:
            return None
        
        data = {}
        data['__prefix__'] = prefix
        for postfix in sample:
            file_name = prefix + '.' + postfix
            d = self.decode(self.tars[file_name], file_name)
            data[postfix] = d
        return data