"""
This module provides resources to load input files and save outputs.

It contains the following functions:

    * load_data
    * save_model
"""

import os
import pandas as pd
import joblib
import sys
sys.path.append('/home/jrondina/Desktop/PycharmProjects/PreProcessing/')
import pre_processing_ssnap
import importlib
importlib.reload(pre_processing_ssnap)

def load_data(dt_src, input_folder, filename):
    """Load data sources and index files

    Parameters
    ----------
    dt_src : str
        Data source (i.e. 'non-img-native', 'non-img-embedding', 'lesion-embedding', 'multi-concatenation',
                            or 'multi-joint-embedding').
    input_folder : str
        Absolute path of the data to be loaded.
    filename : str
        Name of the data file.

    Returns
    -------
    pre_processed_data : Pandas DataFrame
        Pre-processed SSNAP data.

    """

    variables_filename = input_folder + 'IF/voi_list.txt'

    # Obtain list of variables of interest from text file
    var_list = []
    with open(variables_filename, 'r') as fp:
        for line in fp:
            x = line[:-1]
            var_list.append(x)
    print('\nPre-processing file ' + filename + '...')
    pre_processed_data = pre_processing_ssnap.prepare_data(input_folder + filename, var_list)

    if dt_src == 'nonimg-native':
        return pre_processed_data
    if dt_src == "lesion" or dt_src == "multi":
        lesions_encoded = pd.read_csv(limgfilename)
        return dataset_encoded, idx, lesions_encoded


def save_model(md, ev, xts, yts, alg, opt, dt_src, stype, stage, cln, imb_st, cv, path):
    """Save trained model

    Parameters
    ----------
    md : object
        Trained model
    ev : str
        Title of the event
    xts : Pandas Dataframe
        Test dataset
    yts : Pandas Dataframe
        Test labels
    alg : str
        Name of the algorithm used to build the model
    dt_src : str
        Data source
    stype : str
        Stroke type
    stage : str
        Stage when predictors where collected (e.g. 72hours)
    cln : str
        Label noise cleaning (or 'none')
    path : str
        Absolute path of folder where results will be saved.


    Returns
    -------
    filename : str
        Name of the file contained the trained model.
    """

    filename = path + 'model_' + alg + '_' + opt + '_' + dt_src + '_' + stype + '_' + stage + '_' + cln + '_' + imb_st + '_' + cv + '_' + ev

    joblib.dump(md, (filename + '.pkl'))
    xts.to_csv((filename.replace('model', 'x_test') + '.csv'))
    yts.to_csv((filename.replace('model', ' y_test') + '.csv'))
    yts.to_csv((filename.replace('model', ' y_test') + '.csv'))

    return filename
