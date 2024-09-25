import pandas as pd

# BPD                   : 2022-11-16T11:00:00 - 2024-05-14T09:15:00
# sdoml-lite-biosentinel: 2022-11-01T00:00:00 - 2024-05-14T19:30:00
# CRaTER-D1D2           : 2009-06-26T21:00:00 - 2024-06-19T00:00:00

events = []
for y in range(15):
    events.append(('test','20%s-05-01T00:00:00'%(y+10), '20%s-05-31T23:59:59'%(y+10)))
    events.append(('valid','20%s-03-01T00:00:00'%(y+10), '20%s-03-31T23:59:59'%(y+10)))
events = pd.DataFrame(events, columns=['prefix', 'date_start', 'date_end',])
format = '%Y-%m-%dT%H:%M:%S'
events['date_start'] = pd.to_datetime(events['date_start'], format=format)
events['date_end'] = pd.to_datetime(events['date_end'], format=format)

EventCatalog = {}
for prefix in events['prefix'].unique():
    events_with_prefix = events[events['prefix'] == prefix]
    num_events = len(events_with_prefix)
    for i in range(num_events):
        event = events_with_prefix.iloc[i]
        event_id = prefix + str(i+1).zfill(len(str(num_events)))
        date_start = event['date_start'].isoformat()
        date_end = event['date_end'].isoformat()
        EventCatalog[event_id] = date_start, date_end

#print(EventCatalog)
#for event, val in EventCatalog.items():
#    print(event, val[0], val[1])

# event_id      date_start          date_end            max_pfu
# test01 2010-05-01T00:00:00 2010-05-31T23:59:59
# test02 2011-05-01T00:00:00 2011-05-31T23:59:59
# test03 2012-05-01T00:00:00 2012-05-31T23:59:59
# test04 2013-05-01T00:00:00 2013-05-31T23:59:59
# test05 2014-05-01T00:00:00 2014-05-31T23:59:59
# test06 2015-05-01T00:00:00 2015-05-31T23:59:59
# test07 2016-05-01T00:00:00 2016-05-31T23:59:59
# test08 2017-05-01T00:00:00 2017-05-31T23:59:59
# test09 2018-05-01T00:00:00 2018-05-31T23:59:59
# test10 2019-05-01T00:00:00 2019-05-31T23:59:59
# test11 2020-05-01T00:00:00 2020-05-31T23:59:59
# test12 2021-05-01T00:00:00 2021-05-31T23:59:59
# test13 2022-05-01T00:00:00 2022-05-31T23:59:59
# test14 2023-05-01T00:00:00 2023-05-31T23:59:59
# test15 2024-05-01T00:00:00 2024-05-31T23:59:59
# valid01 2010-03-01T00:00:00 2010-03-31T23:59:59
# valid02 2011-03-01T00:00:00 2011-03-31T23:59:59
# valid03 2012-03-01T00:00:00 2012-03-31T23:59:59
# valid04 2013-03-01T00:00:00 2013-03-31T23:59:59
# valid05 2014-03-01T00:00:00 2014-03-31T23:59:59
# valid06 2015-03-01T00:00:00 2015-03-31T23:59:59
# valid07 2016-03-01T00:00:00 2016-03-31T23:59:59
# valid08 2017-03-01T00:00:00 2017-03-31T23:59:59
# valid09 2018-03-01T00:00:00 2018-03-31T23:59:59
# valid10 2019-03-01T00:00:00 2019-03-31T23:59:59
# valid11 2020-03-01T00:00:00 2020-03-31T23:59:59
# valid12 2021-03-01T00:00:00 2021-03-31T23:59:59
# valid13 2022-03-01T00:00:00 2022-03-31T23:59:59
# valid14 2023-03-01T00:00:00 2023-03-31T23:59:59
# valid15 2024-03-01T00:00:00 2024-03-31T23:59:59