import sqlite3 as sqlite
import numpy as np
import pickle
from tools import load_spec, load_radar


def initiate_db(db_name: str):
    conn = sqlite.connect(db_name)
    c = conn.cursor()
    c.execute(
        ''' CREATE TABLE IF NOT EXISTS
            data(label TEXT, run INT, raw_spec BLOB, raw_radar BLOB)'''
    )
    return c, conn

def close_db(db_name):
    conn = sqlite.connect(db_name)
    c = conn.cursor()
    c.close()
    conn.close()


def write_entry(db_name: str, label: str, run: int, raw_spec=None, raw_radar=None):
    """ 
    Create new entry in database: (db_name, label, run, spectrum, radar)

    db_name    database name, including extension
    label      str, to record data with. This is the 'target' for learning.
    run        int, data collection run number, along with label, this will uniquely identify an entry.
    raw_spec   numpy array
    raw_radar  numpy array
    """
    conn = sqlite.connect(db_name)
    c = conn.cursor()

    # pickle raw_spec and put it in database
    if raw_spec is not None:
        raw_spec = pickle.dumps(raw_spec)

        c.execute(
            'UPDATE data SET raw_spec=? WHERE label=? AND run=?',
            (raw_spec, label, run)
        )
        c.execute(
            'INSERT INTO data(label, run, raw_spec) SELECT ?,?,? WHERE (Select Changes()=0)',
            (label, run, raw_spec)
        )

    # if raw_radar given;
    if raw_radar is not None:
        raw_radar = pickle.dumps(raw_radar)

        c.execute(
            'UPDATE data SET raw_radar=? WHERE label=? AND run=?',
            (raw_radar, label, run)
        )
        c.execute(
            'INSERT INTO data(label, run, raw_radar) SELECT ?,?,? WHERE (Select Changes()=0)',
            (label, run, raw_radar)
        )

    conn.commit()
    c.close()


def read_from_db(db_name: str, label: str, run: int):
    """
    Returns data in database for given label and run number.

    label  str
    run    int
    """

    # connect to db
    conn = sqlite.connect(db_name)
    c = conn.cursor()
    c.execute('SELECT * FROM data WHERE label = ? AND run = ?', (label, run))

    # fetch data
    row = c.fetchone()

    # convert to python usable format
    row_data = []
    for i in range(len(row)):

        if i <= 1:
            # import label and run 'as is'.
            row_data.append(row[i])
        else:
            if row[i] is None:
                row_data.append(None)
            else:
                # unpickle the data to get back numpy arrays
                row_data.append(pickle.loads(row[i]))
    c.close()
    return row_data


def write_sample(db_name, label, path, spec_list, radar_list, run_start_n=0):
    """
    NEEDS TESTING:
    REQUIRES data to be in path folder, under radar_data and spec_data.
    Possibly make filenames the same across both sides so there would be no need for two input lists here.

    Use: write all data with a given label into database.
    Writes arrays within given sample files into database, alongside the label.
    NOTE: Each file must contain a single measurement (not hi-speed 100).

    label         str, 'aspirin'
    path          str, path to folder where data lives
    spec_list     list of str, ['as0','as1','as2','as3','as4'] 
    radar_list    list of str, corresponding radar data, ['r-as0','r-as1','r-as2','r-as3','r-as4']
    run_start_n   integer, starting sample number for these files. Default 0.
    """

    if len(spec_list) != len(radar_list):
        return 'Error: Number of spectrum files does not match number of radar files!'
    else:
        n = len(spec_list)

    for run in range(1, n+1):
           
        ref = run - 1 + run_start_n
        s_filename = spec_list[ref]
        s_frame = load_spec(path+'spec/'+s_filename+'.txt')

        r_filename = radar_list[ref]
        r_frame = load_radar(path+'radar/'+r_filename+'.csv')

        # write each frame into database with run number that identifies it
        write_entry(db_name, label, run, s_frame, r_frame)
