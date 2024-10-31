############################################
# Script for making a hyperopt search grid #
############################################


# imports
import os
import sys
import json
import argparse
import pickle as pkl
from hyperopt import hp

def read_grid_configuration( jsonfile ):
    ### read grid configuration from a json file
    # input arguments:
    # - jsonfile: path to a json file containing a valid grid definition.
    #             the json object should be a list of dicts,
    #             each dict should have the following items:
    #             - variable: name of the ntuple variable to cut on
    #             - cuttype: either min or max
    #                  - hptype: name of a hyperopt range function, e.g. quniform
    #             - minvalue, maxvalue, stepsize
    # output type:
    # dict containing the grid information
    # keys: strings formed as variable + '_' + cuttype
    # values: lists of the form [hyperopt range function, arguments to hyperopt range function]
    with open(jsonfile, 'r') as f:
        jsonobj = json.load(f)
    config = {}
    for el in jsonobj:
        key = '{}_{}'.format(el['variable'],el['cuttype'])
        value = [getattr(hp,el['hptype']), key, el['minvalue'], el['maxvalue'], el['stepsize']]
        config[key] = value
    return config

def make_grid( config ):
    ### make a hyperopt search grid based on a given configuration
    grid = {}
    for key,val in config.items():
        grid[key] = val[0](*val[1:])
    return grid

def make_str( config ):
    ### make an human readable string based on a given configuration
    res = ''
    for key,val in config.items():
        res += '{}: {}('.format(key,val[0])
        for arg in val[1:-1]: res+='{}, '.format(arg)
        res += '{})\n'.format(val[-1])
    res = res.strip('\n')
    return res

if __name__=='__main__':
    
    # get the current grid configuration
    config = read_grid_configuration('hyperopt/grids/ttZ2l_grid.json')

    # make the actual hyperopt grid
    grid = make_grid(config)
    
    # make a human-readable version of the grid
    gridstr = make_str(config)
    print('Found following grid:')
    print(gridstr)

    # pack both in a dict and writ to a pkl file
    out = {'grid':grid, 'description':gridstr}
    outputpkl = os.path.splitext('hyperopt/ttZ2l_grid')[0]+'.pkl'
    with open(outputpkl,'wb') as f:
        pkl.dump(out, f)
    print('Grid written to {}.'.format(outputpkl))
    