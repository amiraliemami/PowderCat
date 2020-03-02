# test database
import numpy as np
from time import time

# custom tools
from tools import load_spec, load_radar
from db_tools import initiate_db, close_db, write_entry, read_from_db


path = '/home/amir/starcat_data/'

# test
raw = load_radar(path+'p4'+'.csv')
len(raw)

# load all text file data into database!
db_name = 'radar_data.db'

# open database
initiate_db(db_name)

def names(abb: str, start_n: int, end_n=None):
    if end_n is None:
        end_n = start_n + 9
    return [abb+str(n) for n in range(start_n, end_n+1)]

paracetamol_filenames = names('p',4)+names('p2_',14)+names('p_test_',24)
aspirin_filenames = names('a1_',34)+names('a2_',44)+names('a_test_',54)
ibuprofen_filenames = names('ib1_',64)+names('ib2_',74)+ names('ib_test_',84)
glucose_filenames = names('glu1_',94)+names('glu2_',104)+names('glu_test_',114)
big_list = [ 
    ['aspirin', aspirin_filenames],
    ['paracetamol', paracetamol_filenames],
    ['ibuprofen', ibuprofen_filenames],
    ['glucose', glucose_filenames],
]

for label, filenames_list in big_list:
    # write into databse
    n = len(filenames_list)

    for run in range(n):
        
        # load all frames in a given file
        filename = filenames_list[run]
        frame = load_radar(path+filename+'.csv')

        # write into db
        write_entry(db_name, label, run, raw_radar = frame)


## NEW DATABASE FOR TEST DATA ###################### !! CHANGES were made to the above on home computer !!

# # load all text file data into database!
# db_name2 = 'radar_test_data.db'

# # open database
# initiate_db(db_name2)
# def names(abb: str, start_n: int, end_n=None):
#     if end_n is None:
#         end_n = start_n + 9
#     return [abb+str(n) for n in range(start_n, end_n+1)]
# paracetamol_test_filenames = names('p_test_',24)
# aspirin_test_filenames = names('a_test_',54)
# ibuprofen_test_filenames = names('ib_test_',84)
# glucose_test_filenames = names('glu_test_',114)
# big_test_list = [ 
#     ['aspirin', aspirin_test_filenames],
#     ['paracetamol', paracetamol_test_filenames],
#     ['ibuprofen', ibuprofen_test_filenames],
#     ['glucose', glucose_test_filenames],
# ]
# for label, filenames_list in big_test_list:
#     # write into databse
#     n = len(filenames_list)

#     for run in range(n):
        
#         # load all frames in a given file
#         filename = filenames_list[run]
#         frame = load_radar(path+filename+'.csv')

#         # write into db
#         write_entry(db_name2, label, run, raw_radar = frame)


# read from database

# NOTE - per entry, we have:
# 0  label
# 1  run
# 2  raw spectrum
# 3  raw radar (all channels)

data = read_from_db(db_name,label='glucose', run=5)
print(data)

#### NOTES: - tooo many sigfigs?  Could reduce database size with less.
####        - do we need to store all processed data? NO. New code only writes raw data.
####        - opening and closing cursor (and initiating database at every write/read call) might be slowing us DOWN...  Use peewee?