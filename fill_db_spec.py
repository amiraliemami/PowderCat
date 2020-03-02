# test database

import numpy as np
from time import time

# custom tools
from tools import load_spec
from db_tools import initiate_db, close_db, write_entry, read_from_db

# import dark and process standard
w = np.loadtxt('/home/amiremami/OneDrive/University/Project/Development/spectrasuite/wavelengths.txt')
d = load_spec('/home/amiremami/OneDrive/University/Project/Development/spectrasuite/dark_and_standard/dark_n5_15ms.txt',ctype = 'single')
s = load_spec('/home/amiremami/OneDrive/University/Project/Development/spectrasuite/dark_and_standard/standard_n5_15ms.txt', ctype = 'single')
sd = s - d
sd = sd - min(sd) + 1
sd = sd/max(sd)

path = '/home/amiremami/OneDrive/University/Project/Development/spectrasuite/3_05_samples/'

# test
raw = load_spec(path+'as0'+'.txt')
len(raw[0])

# load all text file data into database!
db_name = 'raw_data.db'
# open database
initiate_db(db_name)


big_list = [ 
    ['aspirin',['as0','as1','as2','as3','as4']],
    ['paracetamol', ['p0','p1','p2','p3','p4']],
    ['flour', ['f0','f1','f2','f3','f4']],
    ['beechams', ['bee0','bee1','bee2','bee3','bee4']],
    ['ibuprofen', ['ib0','ib1','ib2','ib3','ib4']],
    ['p_bp_20', ['p_bp_20_0','p_bp_20_1','p_bp_20_2','p_bp_20_3','p_bp_20_4']],
    ['p_bp_50', ['p_bp_50_0','p_bp_50_1','p_bp_50_2','p_bp_50_3','p_bp_50_4']],
    ['p_bp_70', ['p_bp_70_0','p_bp_70_1','p_bp_70_2','p_bp_70_3','p_bp_70_4']]
]

for label, filenames_list in big_list:
    # write into databse
    n = len(filenames_list)

    for run in range(n):
        
        # load all frames in a given file
        filename = filenames_list[run]
        frames = load_spec(path+filename+'.txt')

        # get first frame in file (emulate single file)
        frame = frames[0]
        # write into db
        write_entry(db_name, label, run, frames)


# read from database

# NOTE - per entry, we have:
# 0  label
# 1  run
# 2  raw spectrum
# 3  raw radar (all channels)

data = read_from_db(label='aspirin', run=4)
print(data)

#### NOTES: - tooo many sigfigs?  Could reduce database size with less.
####        - do we need to store all processed data? NO. New code only writes raw data.
####        - opening and closing cursor (and initiating database at every write/read call) might be slowing us DOWN...  Use peewee?






### OLD ####################################

big_list = [ 
    ['aspirin',['as0','as1','as2','as3','as4']],
    ['paracetamol', ['p0','p1','p2','p3','p4']],
    ['flour', ['f0','f1','f2','f3','f4']],
    ['beechams', ['bee0','bee1','bee2','bee3','bee4']],
    ['ibuprofen', ['ib0','ib1','ib2','ib3','ib4']],
    ['p_bp_20', ['p_bp_20_0','p_bp_20_1','p_bp_20_2','p_bp_20_3','p_bp_20_4']],
    ['p_bp_50', ['p_bp_50_0','p_bp_50_1','p_bp_50_2','p_bp_50_3','p_bp_50_4']],
    ['p_bp_70', ['p_bp_70_0','p_bp_70_1','p_bp_70_2','p_bp_70_3','p_bp_70_4']]
]

for label, filenames_list in big_list:
    # write into databse
    n = len(filenames_list)

    for run in range(n):
        
        # load all frames in a given file
        filename = filenames_list[run]
        frames = load_spec(path+filename+'.txt')

        # get number of frames in the file
        m = len(frames)

        for i in range(m):

            # write each frame into database with subrun number that identifies it
            write_to_db(label, run, i, frames[i])

### OLD ####################################