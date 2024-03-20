"""
This module provides resources to train models and make predictions.

It contains the following functions:

    * get_predictors_and_target
    * create_xgboost_model
    * create_tabnet_model
    * define_model
"""

import numpy as np
from xgboost import XGBClassifier
from cleanlab.classification import CleanLearning
from sklearn.model_selection import GridSearchCV
# from pytorch_tabnet.tab_model import TabNetClassifier
import math



def get_predictors_and_target(data, label):
    """Obtain predictors and target to predict specific event

    Parameters
    ----------
    data : Pandas Dataframe
        Dataset (Non-imaging variables, already filtered according to stage)
    label: str
        Title of the event prediction

    Returns
    -------
    inpfeat : Pandas Dataframe
        A subset of the non-imaging variables (with potential leaks removed for specific events).
    trgt : Pandas Series
        One-dimensional binary array, where 1 corresponds to the positive class (event)
    """
    print(label)
    inpfeat = data.loc[:, ~data.columns.str.startswith('S7')]
##    inpfeat = inpfeat.loc[:, ~inpfeat.columns.str.startswith('S5')]

    # For some events, addditional leaks are removed
    match label:
        case 'Discharge home':
            trgt = data['S7DischargeType__H'] + data['S7DischargeType__TC'] + data['S7DischargeType__TCN']
        case 'Discharge home or community':
            trgt = data['S7DischargeType__H']
        case 'Death':
            trgt = data['S7DischargeType__D']
        case 'Thromboloysis':
            trgt = data['S2Thrombolysis__Y']
            inpfeat = inpfeat.loc[:, ~inpfeat.columns.str.startswith('S3')]
            inpfeat = inpfeat.loc[:, ~inpfeat.columns.str.startswith('S2Thrombolysis')]
##        case 'Complication: UTI':
##            trgt = data['S5UrinaryTractInfection7Days__Y']
##        case 'Complication: Pneumonia':
##            trgt = data['S5PneumoniaAntibiotics7Days__Y']
        case 'Early supported discharge':
            trgt = data['S7DischargedEsdmt__NS'] + data['S7DischargedEsdmt__SNS']
        case 'AF at discharge':
            trgt = data['S7DischargeAtrialFibrillation__Y']
        case 'Help with ADL':
            trgt = data['S7AdlHelp__Y']

    return inpfeat, trgt


def create_xgboost_model(xtr, ytr, opt, rt, cln):
    """Creates classifier

    Parameters
    ----------
    xtr : Pandas Dataframe
        Training data
    ytr : Pandas Series
        Training labels
    opt : str
        Optimisation: 'no-opt' or 'grid-search'
    rt : Float
        Imbalance rate between classes
    cln : str
        Flag for cleaning noise labels

    Returns
    -------
    xgbmodel: XGBoost model
        Trained classifier
    """

    ##    xgbmodel = XGBClassifier()


    if (opt == 'grid-search'):
        xgbmodel = XGBClassifier(random_state=14)
        params = {
            'max_depth': [3, 6, 9],
            'min_child_weight': [1, 5, 10],
            'n_estimators': [200, 500, 1000],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.5, 1],
            'colsample_bytree': [0.5, 1],
            'scale_pos_weight': [rt, math.log(rt), 1]
        }
        grs_cv = GridSearchCV(
            estimator=xgbmodel,
            param_grid=params,
            scoring='f1',
            cv=5,
            n_jobs=1,
            verbose=3
        )
        grs_cv.fit(xtr, ytr)
        print(" Results from Grid Search ")
        print("\n The best estimator across ALL searched params:\n", grs_cv.best_estimator_)
        print("\n The best score across ALL searched params:\n", grs_cv.best_score_)
        print("\n The best parameters across ALL searched params:\n", grs_cv.best_params_)
        xgbmodel = grs_cv.best_estimator_
    else:
        xgbmodel = XGBClassifier(scale_pos_weight=rt, random_state=14)
        xgbmodel.fit(xtr, ytr, verbose=True)

    if cln == 'labels_noise':
        cl = CleanLearning(xgbmodel)
        _ = cl.fit(xtr, ytr)
        xgbmodel = cl

    return xgbmodel


# def create_tabnet_model(Xtr, ytr, cln):
    # Xtr.fillna(method="ffill", inplace=True)
    # Xtr = Xtr.to_numpy().astype(float)
    # ytr = ytr.to_numpy().astype(float)

    # Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.30, stratify=ytr, random_state=7)

    ##    tbnmodel = TabNetClassifier(optimizer_fn=torch.optim.Adam,
    ##                    optimizer_params=dict(lr=2e-2),
    ##                    scheduler_params={"step_size":10,
    ##                                        "gamma":0.9},
    ##                    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    ##                    mask_type='entmax')

    # tbnmodel = TabNetClassifier(seed=42)

    ##    tbnmodel.fit(Xtr,ytr, eval_set=[(Xtr, ytr),
    ##                (Xval, yval)], eval_name=['train', 'valid'],
    ##                eval_metric=['auc','accuracy'], max_epochs=1000 ,
    ##                patience=50, batch_size=256, virtual_batch_size=128,
    ##                num_workers=0, weights=1, drop_last=False)

    # tbnmodel.fit(Xtr, ytr)

    # return tbnmodel


def define_model(xtr, ytr, alg, opt, imb_st, cln):
    """Define predictive models

    Parameters
    ----------
    xtr : Pandas Dataframe
        Training data
    ytr : Pandas Series
        Training labels
    alg : String
        Algorithm
    imb_st : String
        Strategy to deal with data imbalance: 'scale-pos_imbalance_strategy' or 'no_imbalance_strategy'
    cln : String
        Flag for cleaning noise labels using Cleanlab

    Returns
    -------
    model: object
        Trained model
    n0 : Integer
        Number of cases in class 0
    n1 : Integer
        Number of cases in class 1
    """

    n0 = len(np.where(ytr == 0)[0])
    n1 = len(np.where(ytr == 1)[0])

    if imb_st == 'scale-pos_imbalance_strategy':
        nrate = n0 / n1  # Obtain imbalance rate
    elif imb_st == 'no_imbalance_strategy':
        nrate = 1

    msg = (
        f'\nTraining \'{alg}\' (nrate): \'{nrate}\':\n\n'
    )
    print(msg)

    if alg == 'XGBoost':
        model = create_xgboost_model(xtr, ytr, opt, nrate, cln)

    # if alg == 'TabNet':
    #     model = create_tabnet_model(xtr, ytr, cln)

    return model, n0, n1


