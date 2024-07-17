from datetime import datetime
import numpy as np
import pandas as pd
import struct

__version__ = '0.01.00'

# Constants
GDFNAMELEN = 16                  # Length of the ascii-names
GDFID = 94325877                # ID for GDF

# Data types
t_undef = int('0000', 16)       # Data type not defined
t_ascii = int('0001', 16)       # ASCII character
t_s32 = int('0002', 16)       # Signed long
t_dbl = int('0003', 16)       # Double
t_null = int('0010', 16)       # No data
t_u8 = int('0020', 16)       # Unsigned char
t_s8 = int('0030', 16)       # Signed char
t_u16 = int('0040', 16)       # Unsigned short
t_s16 = int('0050', 16)       # Signed short
t_u32 = int('0060', 16)       # Unsigned long
t_u64 = int('0070', 16)       # Unsigned 64bit int
t_s64 = int('0080', 16)       # Signed 64bit int
t_flt = int('0090', 16)       # Float

# Block types
t_dir = 256      # Directory entry start
t_edir = 512      # Directory entry end
t_sval = 1024      # Single valued
t_arr = 2048      # Array


# Top level gdf readers
def gdftomemory(gdf_file):
    all = []

    def myprocfunc(params, data):
        all.append({'p': params, 'd': data})

    print(readgdf(gdf_file, myprocfunc))
    return all


def gdftopandas(gdf_file):
    params = []
    data = []
    paramid = 1

    def myprocfunc(blockparams, blockdata):
        nonlocal paramid

        temp = blockparams.copy()
        temp['paramid'] = paramid
        params.append(temp)

        temp = blockdata.copy()
        try:
            templen = len(next(iter(temp.values())))
            tempcol = np.full((templen, ), paramid)
            temp['paramid'] = tempcol
        except StopIteration:
            pass
        data.append(temp)

        paramid += 1

    header = readgdf(gdf_file, myprocfunc)

    params = pd.DataFrame(params)
    params.set_index('paramid', inplace=True)

    data = pd.concat([pd.DataFrame(blockdata) for blockdata in data],
                     sort=False)
    data.reset_index(drop=True, inplace=True)

    return header, params, data


# Internal functions
def _parseblocks(f, params, procfunc):
    data = {}
    while True:
        # TODO check end of file?
        # Read block header
        name = f.read(GDFNAMELEN)
        if len(name) == 0:
            procfunc(params, data)
            return
        name = name.decode().rstrip('\x00')
        type_, = struct.unpack('i', f.read(4))
        size, = struct.unpack('i', f.read(4))

        # Get block type and data type
        is_dir = (type_ & t_dir != 0)
        is_edir = (type_ & t_edir != 0)
        is_sval = (type_ & t_sval != 0)
        is_arr = (type_ & t_arr != 0)

        dattype = type_ & 255

        # Get block data (scalar or array)
        if is_sval:
            if dattype == t_dbl:
                value = struct.unpack('d', f.read(8))[0]
            elif dattype == t_null:
                pass
            elif dattype == t_ascii:
                value = f.read(size).decode().rstrip('\x00')
            elif dattype == t_s32:
                value = struct.unpack('i', f.read(4))[0]
            else:
                print('unknown datatype of value!!!')
                print('name=', name)
                print('type=', type_)
                print('size=', size)
                print('dattype=', dattype)
                value = f.read(size)
        elif is_arr:
            if dattype == t_dbl:
                value = np.frombuffer(f.read(size))
                data[name] = value
            else:
                print('unknown datatype of value!!!')
                print('name=', name)
                print('type=', type_)
                print('size=', size)
                print('dattype=', dattype)
                value = f.read(size)

        # Take care of recursion
        if is_dir:
            myparams = params.copy()
            myparams[name] = value
            _parseblocks(f, myparams, procfunc)
        elif is_edir:
            procfunc(params, data)
            return
        elif is_sval:
            params[name] = value


def _proc_print(params, data):
    print(params)
    print(data.keys())
    print('_____________________')


# Base gdf reader
def readgdf(gdf_file, procfunc=_proc_print):
    # Output data
    gdf_head = {}

    with open(gdf_file, 'rb') as f:
        # Read the GDF main header
        gdf_id_check = struct.unpack('i', f.read(4))[0]
        if gdf_id_check != GDFID:
            raise RuntimeWarning('File is not a .gdf file')

        time_created = struct.unpack('i', f.read(4))[0]
        time_created = datetime.fromtimestamp(time_created)
        gdf_head['time_created'] = time_created.isoformat(' ')

        gdf_head['creator'] = f.read(GDFNAMELEN).decode().rstrip('\x00')

        gdf_head['destination'] = f.read(GDFNAMELEN).decode().rstrip('\x00')

        major = struct.unpack('B', f.read(1))[0]
        minor = struct.unpack('B', f.read(1))[0]
        gdf_head['gdf_version'] = str(major) + '.' + str(minor)

        major = struct.unpack('B', f.read(1))[0]
        minor = struct.unpack('B', f.read(1))[0]
        gdf_head['creator_version'] = str(major) + '.' + str(minor)

        major = struct.unpack('B', f.read(1))[0]
        minor = struct.unpack('B', f.read(1))[0]
        gdf_head['destination_version'] = str(major) + '.' + str(minor)

        f.seek(2, 1)   # skip 2 bytes to go to next block

        # Read GDF data blocks (starts recursive reading)
        _parseblocks(f, {}, procfunc)

    return gdf_head
