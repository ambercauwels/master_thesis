#####################################
# Run hyperopt for cut optimization #
#####################################


# imports
from lib2to3.pytree import type_repr
import sys
import os
import argparse
import numpy as np
import awkward as ak
import pickle as pkl
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from functools import partial
import pandas as pd

def pass_selection( events, cuts ):
    ### get a mask for a particular configuration of cuts
    # input arguments:
    # - events: NanoAOD events array
    # - cuts: dictionary mapping a variable name to a cut value
    mask = np.ones(len(events)).astype(bool)
    for cutname, cutvalue in cuts.items():
        # parse cut name
        ismax = cutname.endswith('_max')
        ismin = cutname.endswith('_min')
        if( not (ismax or ismin) ): 
            raise Exception('ERROR: cut {} is neither min nor max.'.format(cutname))
        varname = cutname[:-4]
        # get the variable value
        # todo: update to other scenarios
        varvalue = None
        if varname=='AK4_pt': varvalue = events['AK4_pt']
        elif varname=='AK8_pt': varvalue = events['AK8_pt']
        elif varname=='H_t': varvalue = events['H_t']
        elif varname=='t_tag': varvalue = events['t_tag']
        elif varname=='b_tag': varvalue = events['b_tag']
        #elif varname=='Minv': varvalue = events['Minv']
        else:
            raise Exception('ERROR: variable {} not recognized.'.format(varname))
        # perform the cut
        if ismax: mask = ((mask) & (varvalue < cutvalue))
        if ismin: mask = ((mask) & (varvalue > cutvalue))
    return mask


def calculate_loss(events, cuts,
                    sig_mask=None,
                    lossfunction='negsoverb',
                    iteration=None):
    ### calculate the loss function for a given configuration of cuts

    # print progress
    #print('Now processing iteration {}'.format(iteration[0]))
    iteration[0] += 1

    # do event selection
    sel_mask = pass_selection(events, cuts)
    # calculate number of passing events
    # todo: use sum of relevant weights instead of sum of entries
    nsig_tot = np.sum(events['weights'][sig_mask])
    nsig_pass = np.sum(events['weights'][sig_mask & sel_mask])
    nbkg_tot = np.sum(events['weights'][~sig_mask])
    nbkg_pass = np.sum(events['weights'][~sig_mask & sel_mask])

    # calculate loss
    if lossfunction=='negsoverb':
        if nbkg_pass == 0: loss = 0.
        else: loss = -nsig_pass / np.sqrt(nbkg_pass)
    else:
        msg = 'ERROR: loss function {} not recognized.'.format(lossfunction)
        raise Exception(msg)

    # extend with other loss function definitions
    extra_info = {'nsig_tot': nsig_tot,
                  'nsig_pass': nsig_pass,
                  'nbkg_tot': nbkg_tot,
                  'nbkg_pass': nbkg_pass,
                  'lossfunction': lossfunction}
    return {'loss':loss, 'status':STATUS_OK, 'extra_info': extra_info}


if __name__=='__main__':
    
    inputfile = 'hyperopt/ttZ2l_data.txt'
    gridfile = 'hyperopt/ttZ2l_grid.pkl'
    outputfile= 'hyperopt/ttZ2l_output.pkl'
    niterations= 1000
    nstartup= 100
    
    # load the events array
    events = pd.read_csv(inputfile, sep="\t")
    # define signal mask
    # todo: update to something realistic
    sig_mask = events.signal
    
    # get the grid
    with open(gridfile,'rb') as f:
        obj = pkl.load(f)
        grid = obj['grid']
        gridstr = obj['description']
    print('Found following grid:')
    print(gridstr)

    # run hyperopt
    trials = Trials()
    iteration = [1]
    best = fmin(
      fn=partial(calculate_loss, events,
                 sig_mask=sig_mask,
                 iteration=iteration
      ),
      space=grid,
      algo=partial(tpe.suggest, n_startup_jobs=nstartup),
      max_evals=niterations,
      trials=trials
    )

    # write search results to output file
    if outputfile is not None:
        print('Writing results to {}'.format(outputfile))
        with open(outputfile,'wb') as f:
            pkl.dump(trials,f)
