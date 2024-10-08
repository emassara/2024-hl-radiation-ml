import argparse
import pprint
import sys
import datetime
import time
import os
import urllib.request
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def process(file_names):
    remote_file_name, local_file_name = file_names

    print('Remote: {}'.format(remote_file_name), flush=True)
    os.makedirs(os.path.dirname(local_file_name), exist_ok=True)
    timeout = 15 # seconds
    retries = 5
    for i in range(retries):
        if i > 0:
            print('Retrying ({}/{}): {}'.format(i+1, retries, remote_file_name))
            time.sleep(0.5)
        try:
            r = urllib.request.urlopen(remote_file_name, timeout=timeout)
            open(local_file_name, 'wb').write(r.read())
            print('Local : {}'.format(local_file_name))
            return True
        except Exception as e:
            print('Error ({}): {}'.format(remote_file_name, e))
    if os.path.exists(local_file_name):
        os.remove(local_file_name)
    return False


def main():
    description = 'FDL-X 2024, Radiation Team, Radio Solar Telescope Network (RSTN) Solar Radio Burst data downloader'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--date_start', type=str, default='2009-06-26', help='Start date') # Available from 1981-10-10, but we start from 2009-06-26, beginning of CRaTER data
    parser.add_argument('--date_end', type=str, default='2024-07-29', help='End date')
    parser.add_argument('--remote_root', type=str, default='https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-radio/rstn-1-second/sagamore-hill/', help='Remote root')
    parser.add_argument('--target_dir', type=str, help='Local root', required=True)
    parser.add_argument('--max_workers', type=int, default=1, help='Max workers')
    parser.add_argument('--worker_chunk_size', type=int, default=1, help='Chunk size per worker')
    parser.add_argument('--total_nodes', type=int, default=1, help='Total number of nodes')
    parser.add_argument('--node_index', type=int, default=0, help='Node index')
    
    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    date_start = datetime.datetime.fromisoformat(args.date_start)
    date_end = datetime.datetime.fromisoformat(args.date_end)

    current = date_start
    file_names = []
    while current < date_end:
        # Sample URL
        # https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-radio/rstn-1-second/sagamore-hill/2023/01/08jan23.k7o.gz
        # https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-radio/rstn-1-second/sagamore-hill/2010/05/05MAY10.K7O.gz
        # 2009: up
        # 2010: up
        # 2011: up
        # 2012: up
        # 2013: down
        # 2014: down
        # 2015: down
        # 2016: down
        # 2017: up
        # 2018: up
        # 2019: up
        # 2020: down
        # 2021: down
        # 2022: down
        # 2023: down
        # 2024: down

        file_name_down = '{:%d%b%y}.k7o'.format(current).lower() + '.gz'
        file_name_up   = '{:%d%b%y}.k7o'.format(current).upper() + '.gz'

        if current.year in [2009, 2010, 2011, 2012, 2017, 2018, 2019]:
            file_name = file_name_up
        else:
            file_name = file_name_down

        remote_file_name = os.path.join(args.remote_root, '{:%Y/%m}'.format(current), file_name)
        local_file_name = os.path.join(args.target_dir, '{:%Y/%m}'.format(current), file_name)
        file_names.append((remote_file_name, local_file_name))

        current += datetime.timedelta(days=1)


    if len(file_names) == 0:
        print('No files to download.')
        return
    
    if len(file_names) < args.total_nodes:
        print('Total number of files is less than the total number of nodes.')
        return

    files_per_node = len(file_names) // args.total_nodes
    # get the subset of file names for this node, based on the total number of nodes and the node index
    file_names_for_this_node = file_names[args.node_index * files_per_node : (args.node_index + 1) * files_per_node]
    
    print('Total nodes: {}'.format(args.total_nodes))
    print('Node index : {}'.format(args.node_index))
    print('Total files for all nodes : {}'.format(len(file_names)))
    print('Total files for this node : {}'.format(len(file_names_for_this_node)))
    
    if args.max_workers == 1:
        results = list(map(process, file_names_for_this_node))
    else:
        results = process_map(process, file_names_for_this_node, max_workers=args.max_workers, chunksize=args.worker_chunk_size)

    print('Files downloaded: {}'.format(results.count(True)))
    print('Files skipped   : {}'.format(results.count(False)))
    print('Files total     : {}'.format(len(results)))
    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()