"""
This module provides resources to evaluate trained models, display accuracy measures and plot graphs.

It contains the following functions:

    * print_model_performance
    * print_shap
    * get_potential_label_errors
    * print_info_noisy_labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score
import shap
# from cleanlab.filter import find_label_issues
# from xgboost import XGBClassifier
# from sklearn.model_selection import cross_val_predict


def print_model_performance(ev, mdl, filename, dt_src, alg):
    """Obtain prediction performance metrics, display accuracy measures on the screen and save accuracy plots.
    The plots are saved in the folder FIGURES, with the same name of the model with prefixes determined by the
    type of the plot courve plotted.

    Parameters
    ----------
    ev : str
        Title of the event
    mdl : object
        Trained classifier
    filename : string
        Name of file containing a trained model
    dt_src : str
        Data source
    alg : str
        Algorithm used to train model


    Returns
    -------
    Returns f1-score
    """

    xts = pd.read_csv(filename.replace('model', 'x_test') + '.csv', index_col=0)

    # Make predictions for test data
    if alg == 'XGBoost':
        y_pred = mdl.predict(xts)
        y_probpred = mdl.predict_proba(xts)[:, 1]

    if alg == 'TabNet':
        xts.fillna(method="ffill", inplace=True)
        xts = xts.to_numpy().astype(float)
        y_pred = mdl.predict(xts)
        y_probpred = mdl.predict_proba(xts)[:, 1]

    yts = pd.read_csv(filename.replace('model', ' y_test') + '.csv', index_col=0)

    nfeatures = len(xts.columns)
    cmatrix = confusion_matrix(yts, y_pred)
    cl_acc = cmatrix.diagonal() / cmatrix.sum(axis=1)
    aucroc = metrics.roc_auc_score(yts, y_probpred)
    aucpr = metrics.average_precision_score(yts, y_probpred)
    f1 = f1_score(yts, y_pred)
    print(yts)
    print(type(yts))
    print(nfeatures)

    n1 = sum(yts.iloc[:,0])
    n0 = len(yts)-n1

    # Print in console
    msg = (
        f'\nPredict \'{ev}\' from \'{dt_src}\' ({nfeatures} features) using \'{alg}\':\n\n'
        f'N cases per class in the test set: {n0} and {n1}\n'
        f'Accuracy per class: {cl_acc[0]:.3f} and {cl_acc[1]:.3f}\n'
        f'Balanced accuracy: {(cl_acc[0] + cl_acc[1]) / 2:.3f}\n'
        f'AUC ROC: {aucroc:.3f}\n'
        f'AUC PR: {aucpr:.3f}\n'
        f'F1: {f1:.3f}\n\n'
    )
    print(msg)

    msg_latex = (
        f'\n {ev} & {n0} X {n1} & '
        f'{cl_acc[0]:.3f} & {cl_acc[1]:.3f} & '
        f'{aucroc:.3f} & {aucpr:.3f} & {f1:.3f} \n\n'
    )
    print(msg_latex)

    # Plot ROC, PR, and calibration courves

    fpr, tpr, _ = metrics.roc_curve(yts, y_probpred)
    plt.figure(1)
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(ev)
    plt.savefig(filename.replace('model', 'FIGURES/roc_courve') + '.png')
    plt.clf()

    precision, recall, thresholds = precision_recall_curve(yts, y_probpred)
    plt.figure(2)
    plt.plot(recall, precision)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title(ev)
    plt.savefig(filename.replace('model', 'FIGURES/pr_courve') + '.png')
    plt.clf()

    x, y = calibration_curve(yts, y_probpred, n_bins=10)
    plt.figure(3)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Ideally Calibrated')  # Plot perfectly calibrated
    plt.plot(y, x, marker='.', label='XGB Classifier')  # Plot model's calibration curve
    plt.legend(loc='upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.title(ev)
    plt.savefig(filename.replace('model', 'FIGURES/calibration_courve') + '.png')
    plt.clf()

    f1 = f1_score(yts, y_pred)
    return cmatrix, f1, aucpr, y_pred, y_probpred


def plot_accuracy_on_embedding(inp_feat_ev, yt, yp, yprobp, ev, mode, filename):
    """Plot embedding with colors based on the accuracy of prediction for each case in test set.
    The following code is used: 0 (true negative); 1 (false positive); 2 (false negative); 3 (true positve)

       Parameters
       ----------
       yt : array
           Targets of test set
       yp : array
           Predicted targets for test set

    Returns
    -------
    Save figure
    """

    aux = yt.to_numpy() * 2
    code = aux + yp
    
    ind0 = np.where(code == 0)
    ind1 = np.where(code == 1)
    ind2 = np.where(code == 2)
    ind3 = np.where(code == 3)

    embed = pd.read_csv('/home/jrondina/Desktop/PycharmProjects/ssnap-representations/RESULTS/ae_2D__batch_norm__eta3__long_run/embeddings_kch_epoch_18350.csv')
    clust = pd.read_csv('/home/jrondina/Desktop/PycharmProjects/clustering/RESULTS/ae_2D__batch_norm__eta3__long_run/labels__ae_2D__batch_norm__eta3__long_run__epoch_18350__manual_16labels__test_data_KDtrees.csv')
    data_test = pd.read_csv('/home/jrondina/Desktop/PycharmProjects/ssnap-representations/RESULTS/ae_2D__batch_norm__eta3__long_run/data_test.csv')

    """
    Adjust indices
    """
    aux1 = inp_feat_ev.index
    aux2 = data_test['Unnamed: 0']
    np_aux1 = aux1.to_numpy()
    np_aux2 = aux2.to_numpy()
    aux = np.intersect1d(np_aux1,np_aux2)
    final_indices=np.nonzero(np.isin(np_aux2,aux))[0]

    filtered_embed = embed.iloc[final_indices,:]
    filtered_embed = filtered_embed.reset_index()

    filtered_clust = clust.iloc[final_indices,:]
    filtered_clust = filtered_clust.reset_index()

    plt.figure(4)
    sz = 2
    
    if mode == 'Individual accuracy':
        sct0 = plt.scatter(filtered_embed.loc[ind0,'0'], filtered_embed.loc[ind0,'1'], s = sz, c = 'lightsteelblue')
        sct1 = plt.scatter(filtered_embed.loc[ind1,'0'], filtered_embed.loc[ind1,'1'], s = sz, c = 'darkred')
        sct2 = plt.scatter(filtered_embed.loc[ind2,'0'], filtered_embed.loc[ind2,'1'], s = sz, c = 'darkblue')
        sct3 = plt.scatter(filtered_embed.loc[ind3,'0'], filtered_embed.loc[ind3,'1'], s = sz, c = 'rosybrown')
        plt.legend((sct0, sct1, sct2, sct3), ('True Negative', 'False Positive', 'False Negative', 'True Positive'), scatterpoints=1, fontsize=8, markerscale=5)

    if mode == 'AUC-PR per cluster' or mode == 'F1 per cluster':      
        labels = np.unique(filtered_clust['0'].to_numpy())
        ytest = yt.to_numpy()
        for ind in labels:            
            lab_ind = filtered_clust.index[filtered_clust['0'] == ind].tolist()
            if ind > -1:              
                if mode == 'AUC-PR per cluster':
                    cl_acc = metrics.average_precision_score(ytest[lab_ind], yprobp[lab_ind])
                if mode == 'F1 per cluster':
                    cl_acc = f1_score(ytest[lab_ind], yp[lab_ind])
            else:
                cl_acc = 0
            filtered_embed.loc[lab_ind,'cl_acc'] = cl_acc
        sc = plt.scatter(filtered_embed.loc[:,'0'], filtered_embed.loc[:,'1'], s=sz, c=filtered_embed.loc[:,'cl_acc'], cmap='gray')
        plt.colorbar(sc)

    plt.title('Prediction of ' + ev + ' (' + mode + ')', wrap=True)
    plt.savefig(filename.replace('model', 'FIGURES/acc_on_embedding__' + mode) + '.png')
    plt.clf()
    
    return filtered_embed, ind0, filtered_clust
    

def print_shap(ev, mdl, xtr, filename):
    """Plot shap summary

    Parameters
    ----------
    ev : str
        Title of the event
    mdl : object
        Trained classifier
    xtr : Pandas dataframe
        Training data
    filename : str
        Name of file containing a trained model

    Returns
    -------
    Display plot on the screen and save them in output file.
    """

    explainer = shap.TreeExplainer(mdl)
    shap_values = explainer.shap_values(xtr)
    plt.figure()
    shap.summary_plot(shap_values, xtr, show=False)
    plt.savefig(filename.replace('model', 'FIGURES/shap_summary') + '.png', bbox_inches='tight')
    plt.clf()


# def get_potential_label_errors(X, y, n0, n1, n_est=200):
#     """Obtain potential label errors using Cleanlab
#
#     Parameters
#     ----------
#     X : Pandas Dataframe
#     y : Pandas Series
#         One-dimensional binary array, where 1 corresponds to the positive class (event)
#     n0 : integer
#         Number of cases in class 0
#     n1 : integer
#         Number of cases in class 1
#
#     Returns
#     -------
#     ranked_label_issues :numpy array
#         Indices of the identified label issues sorted by cleanlab’s self-confidence score
#     """
#
#     nrate = max(n0, n1) / min(n0, n1)
#     xgbmodel = XGBClassifier(scale_pos_weight=nrate, n_estimators=n_est, eval_metric='aucpr', random_state=14)
#
#     num_crossval_folds = 3
#     pred_probs = cross_val_predict(
#         xgbmodel, X, y, cv=num_crossval_folds, method='predict_proba',
#     )
#
#     ranked_label_issues = find_label_issues(
#         labels=y, pred_probs=pred_probs, return_indices_ranked_by='self_confidence'
#     )
#
#     return ranked_label_issues


# def print_info_noisy_labels(ranked_label_issues, data, labels, filename, dt_src):
#     """Print features extracted from the cases  potential label errors using Cleanlab
#
#     Parameters
#     ----------
#     ranked_label_issues :numpy array
#         Indices of the identified label issues sorted by cleanlab’s self-confidence score
#     data : Pandas Dataframe
#         Input features
#     labels : Pandas Series
#         One-dimensional binary array, where 1 corresponds to the positive class (event)
#     filename : str
#         Name of file containing a trained model
#
#     Returns
#     -------
#     Display number of cases removed on the screen and save information about them in output file.
#     """
#
#     n_top = round(len(ranked_label_issues) * 0.1)
#     top_indices = ranked_label_issues[:n_top]
#
#     msg = (
#         f'Number of cases removed by cleanlab: {n_top} \n\n'
#     )
#     print(msg)
#
#     dt = data
#     dt.insert(0, 'Label', labels)
#
#     data_info = dt.iloc[top_indices, :]
#
#     if dt_src != 'non-img':
#         # Print info from images
#         print('include imaging info')
#
#     fname = filename.replace('model', 'info_cleanlab') + '.csv'
#     data_info.to_csv(fname)
