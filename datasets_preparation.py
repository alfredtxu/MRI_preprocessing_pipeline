"""
This module contains code to prepare datasets for subsequent analysis.

It contains the following functions:

    *get_targets
    
"""

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
import importlib
import pre_processing_ssnap
import modeling
import input_output
import joblib
import evaluation
importlib.reload(pre_processing_ssnap)
importlib.reload(modeling)
importlib.reload(input_output)
importlib.reload(evaluation)


def get_targets(dt, ev, sind):
    """
    Obtain arrays of binary values corresponding to targets to be predicted.

    Parameters
    ----------
    dt : Pandas DataFrame
        Pre-processed ssnap-dataset
    ev : str
        Title of the event to be predicted
    sind : Pandas Series
        SSNAP indices for specific analysis

    Returns
    -------
    Pandas Series
        Series of binary values
        
    """

    match ev:

        case 'dischargeHome':        
            trgt = pd.get_dummies(dt.S7DischargeType).H
            target = trgt.iloc[sind]

        case 'dischargeHomeOrCommunity':
            trgt = pd.get_dummies(dt.S7DischargeType).H + pd.get_dummies(dt.S7DischargeType).TC + pd.get_dummies(dt.S7DischargeType).TCN
            target = trgt.iloc[sind]

        case 'death':
            trgt = pd.get_dummies(dt.S7DischargeType).D
            target = trgt.iloc[sind]

        case 'AFatDischarge':
            trgt = pd.get_dummies(dt.S7DischargeAtrialFibrillation).Y
            target = trgt.iloc[sind]

        case 'complicationUTI':
            trgt = pd.get_dummies(dt.S5UrinaryTractInfection7Days).Y
            target = trgt.iloc[sind]

        case 'complicationPneumonia':
            trgt = pd.get_dummies(dt.S5PneumoniaAntibiotics7Days).Y
            target = trgt.iloc[sind]

        case 'ESD':
            trgt = pd.get_dummies(dt.S7DischargedEsdmt).NS + pd.get_dummies(dt.S7DischargedEsdmt).SNS
            target = trgt.iloc[sind]

        case 'helpWithADL':
            trgt = pd.get_dummies(dt.S7AdlHelp).Y
            target = trgt.iloc[sind]

        case 'thromboloysis':
            trgt = pd.get_dummies(dt.S2Thrombolysis).Y
            target = trgt.iloc[sind]

        case 'psychology':
            trgt = pd.get_dummies(dt.S4Psychology).Y
            target = trgt.iloc[sind]

        case 'physiotherapy':
            trgt = pd.get_dummies(dt.S4Physio).Y
            target = trgt.iloc[sind]

        case 'occupationalTherapy':
            trgt = pd.get_dummies(dt.S4OccTher).Y
            target = trgt.iloc[sind]

        case 'SpeechLangTherapy':
            trgt = pd.get_dummies(dt.S4SpeechLang).Y
            target = trgt.iloc[sind]
           
    return target


if __name__ == "__main__":

    data_source = 'combinedEmbeddedLesionsNI'
    stroke_type = 'ischaemic'
    
    timeframe = 'prospective'
    cv_type = 'intra-site'
    predictors_stage = '48hours'
    algorithm = 'XGBoost'
    optimisation = 'grid-search'
    imbalance_strategy = 'scale-pos_imbalance_strategy'
    cleaning = 'no-cleaning'

    data_path = '/media/jrondina/MPBRCII/Jane/'
    output_path = '/home/jrondina/Desktop/PycharmProjects/predictive-tool/RESULTS/'
    datasets_path = '/home/jrondina/Desktop/PycharmProjects/predictive-tool/DATASETS/'

    filename_NI_retrospective_site1 = data_path + 'SSNAP_UCLH_retrospective.xlsx'
    filename_NI_retrospective_prospective_site2 = data_path + 'ssnap_dwi-ischaemic_lesseg.csv'
    
##    filename_NI_prospective_site2
##    filename_embeddedNI_retrospective_site1
##    filename_embeddedNI_retrospective_site2
##    filename_embeddedNI_prospective_site1
##    filename_embeddedNI_prospecive_site2
##    filename_embeddedLES_retrospective_site1
##    filename_embeddedLES_retrospective_site2
##    filename_embeddedLES_prospective_site1
##    filename_embeddedLES_prospecive_site2
##    filename_clusters_retrospective_site1
##    filename_clusters_retrospective_site2
##    filename_clusters_prospective_site1
##    filename_clusters_prospecive_site2

    events = ['dischargeHome', 'dischargeHomeOrCommunity', 'death', 'AFatDischarge', 'complicationUTI', 'complicationPneumonia', 'ESD', 'helpWithADL', 'thromboloysis', 'psychology', 'physiotherapy', 'occupationalTherapy', 'SpeechLangTherapy']
##    events = ['ESD']
##    events = ['thromboloysis', 'psychology', 'physiotherapy', 'occupationalTherapy', 'SpeechLangTherapy']


    
    if cv_type == 'intra-site':

        if timeframe == 'retrospective':

            indexing_file = 'matching__UCLH__SSNAP__retrospective__to__LESIONS.csv'

            """
            Pre-process SSNAP data
            """
            
            variables_filename = datasets_path + 'voi_list.txt'

            # Obtain list of variables of interest from text file
            var_list = []
            with open(variables_filename, 'r') as fp:
                for line in fp:
                    x = line[:-1]
                    var_list.append(x)
                    
            dataset_NI_retrospective_site1 = pre_processing_ssnap.prepare_data(filename_NI_retrospective_site1, var_list)

            indexing = pd.read_csv(datasets_path + 'SEGMENTATIONS/RETROSPECTIVE_UCLH_SSNAP_LESIONS/matching__UCLH__SSNAP__retrospective__to__LESIONS.csv')

            """
            Pre-process SSNAP data
            """
            
            variables_filename = datasets_path + 'voi_list.txt'

            # Obtain list of variables of interest from text file
            var_list = []
            with open(variables_filename, 'r') as fp:
                for line in fp:
                    x = line[:-1]
                    var_list.append(x)
                    
            dataset_NI_retrospective_site1 = pre_processing_ssnap.prepare_data(filename_NI_retrospective_site1, var_list)

            indexing = pd.read_csv(datasets_path + indexing_file)
            ssnap_indices = indexing['ssnapIndex']
##            ssnap_indices = indexing.loc[indexing['imgValidFlag'] == 1, 'ssnapIndex']indexing_file)
            ssnap_indices = indexing['ssnapIndex']
##            ssnap_indices = indexing.loc[indexing['imgValidFlag'] == 1, 'ssnapIndex']


            """
            Create datasets for data source or combination
            """
            
            if data_source == 'nativeNI':
            
                ssnap_subset = dataset_NI_retrospective_site1.iloc[ssnap_indices,:]  
                pp_ssnap_subset = pre_processing_ssnap.agglomerate_ethnic_groups(ssnap_subset)
                pp_ssnap_subset = pre_processing_ssnap.select_variables(pp_ssnap_subset, predictors_stage)
                pp_ssnap_subset = pd.get_dummies(pp_ssnap_subset, columns=pp_ssnap_subset.select_dtypes('object').columns, prefix_sep='__')
                ni_features = pp_ssnap_subset.loc[:, ~pp_ssnap_subset.columns.str.startswith('S7')]
                       
                for i, event in enumerate(events):

                    target = get_targets(dataset_NI_retrospective_site1, event, ssnap_indices)
                    
                    if event == 'thromboloysis':          
                        reduced_ni_features = ni_features.loc[:, ~ni_features.columns.str.startswith('S2Thrombolysis')]
                    
                    train_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '__' + event + '__' + 'trainData.csv'
                    test_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '__' + event + '__' + 'testData_.csv'
                    train_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '__' + event + '__' + 'trainTargets.csv'
                    test_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '__' + event + '__' + 'testTargets.csv'
                     
                    X_train, X_test, y_train, y_test = train_test_split(ni_features, target, test_size=0.30, stratify=target.to_numpy(), random_state=7)
                    if event == 'thromboloysis': 
                        X_train, X_test, y_train, y_test = train_test_split(reduced_ni_features, target, test_size=0.30, stratify=target.to_numpy(), random_state=7)
                        
                    pd.DataFrame(X_train).to_csv(train_data_filename)
                    pd.DataFrame(X_test).to_csv(test_data_filename)
                    pd.DataFrame(y_train).to_csv(train_targets_filename)
                    pd.DataFrame(y_test).to_csv(test_targets_filename)

                    
            if data_source == 'embeddedLesions':

                lesion_file = 'LESION_voxels4mm_fwhm4.csv'
                ncomponents = 80
                lesion_data = pd.read_csv(datasets_path + lesion_file)          
      
                lesion_features = lesion_data.drop(['imgKey', 'ssnapIndex'], axis=1)
                
                for i, event in enumerate(events):
                    
                    target = get_targets(dataset_NI_retrospective_site1, event, ssnap_indices)
                    
                    train_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'trainData.csv'
                    test_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'testData_.csv'
                    train_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'trainTargets.csv'
                    test_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'testTargets.csv'              
            
                    X_train, X_test, y_train, y_test = train_test_split(lesion_features, target, test_size=0.30, stratify=target.to_numpy(), random_state=7)

                    nmf_model = NMF(n_components = ncomponents, random_state=0)
                    nmf_model.fit(X_train)
##                    nmf_features = nmf_model.transform(lesion_features)

                    X_train_nmf = nmf_model.transform(X_train)
                    X_test_nmf = nmf_model.transform(X_test)

                    pd.DataFrame(X_train_nmf).to_csv(train_data_filename)
                    pd.DataFrame(X_test_nmf).to_csv(test_data_filename)
                    pd.DataFrame(y_train).to_csv(train_targets_filename)
                    pd.DataFrame(y_test).to_csv(test_targets_filename)


            if data_source == 'combinedEmbeddedLesionsNI':
                
                lesion_file = 'LESION_voxels4mm_fwhm4.csv'
                ncomponents = 80
                lesion_data = pd.read_csv(datasets_path + lesion_file)          
      
                lesion_features = lesion_data.drop(['imgKey', 'ssnapIndex'], axis=1)

                ssnap_subset = dataset_NI_retrospective_site1.iloc[ssnap_indices,:]  
                pp_ssnap_subset = pre_processing_ssnap.agglomerate_ethnic_groups(ssnap_subset)
                pp_ssnap_subset = pre_processing_ssnap.select_variables(pp_ssnap_subset, predictors_stage)
                pp_ssnap_subset = pd.get_dummies(pp_ssnap_subset, columns=pp_ssnap_subset.select_dtypes('object').columns, prefix_sep='__')
                ni_features = pp_ssnap_subset.loc[:, ~pp_ssnap_subset.columns.str.startswith('S7')]
                ni_features = ni_features.reset_index(drop=True)

                for i, event in enumerate(events):
                    
                    target = get_targets(dataset_NI_retrospective_site1, event, ssnap_indices)
##                    combined_features = pd.concat([ni_features.reset_index(), lesion_features], axis=1)
##                  
                    train_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'trainData.csv'
                    test_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'testData_.csv'
                    train_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'trainTargets.csv'
                    test_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'testTargets.csv'
                           
                    X_train, X_test, y_train, y_test = train_test_split(ni_features, target, test_size=0.30, stratify=target.to_numpy(), random_state=7)
                    if event == 'thromboloysis':
                        reduced_ni_features = ni_features.loc[:, ~ni_features.columns.str.startswith('S2Thrombolysis')]
                        X_train, X_test, y_train, y_test = train_test_split(reduced_ni_features, target, test_size=0.30, stratify=target.to_numpy(), random_state=7)
                    
                    nmf_model = NMF(n_components = ncomponents, random_state=0)

                    L_train = lesion_features.loc[X_train.index]
                    L_test = lesion_features.loc[X_test.index]

                    nmf_model.fit(L_train)

                    L_train_nmf = nmf_model.transform(L_train)
                    L_test_nmf = nmf_model.transform(L_test)

                    X_train_nmf = pd.concat([X_train.reset_index(), pd.DataFrame(L_train_nmf)], axis = 1)
                    X_test_nmf = pd.concat([X_test.reset_index(), pd.DataFrame(L_test_nmf)], axis = 1)

                    print(event)
                    print(X_train_nmf.shape)

##                    if event == 'thromboloysis':          
##                        reduced_ni_features = ni_features.loc[:, ~ni_features.columns.str.startswith('S2Thrombolysis')]
##                        combined_features = pd.concat([reduced_ni_features.reset_index(), lesion_features], axis=1)
##                    
                    pd.DataFrame(X_train_nmf).to_csv(train_data_filename)
                    pd.DataFrame(X_test_nmf).to_csv(test_data_filename)
                    pd.DataFrame(y_train).to_csv(train_targets_filename)
                    pd.DataFrame(y_test).to_csv(test_targets_filename)
                    

        if timeframe == 'prospective':

            filename_NI = data_path + 'ssnap_dwi-ischaemic_lesseg.csv'
            indexing_file = 'matching__KCH__SSNAP__retrospective_and_prospective__to__LESIONS.csv'

            """
            Pre-process SSNAP data
            """
            
            variables_filename = datasets_path + 'voi_list.txt'

            # Obtain list of variables of interest from text file
            var_list = []
            with open(variables_filename, 'r') as fp:
                for line in fp:
                    x = line[:-1]
                    var_list.append(x)
                    
            dataset_NI = pre_processing_ssnap.prepare_data(filename_NI, var_list)

            indexing = pd.read_csv(datasets_path + 'SEGMENTATIONS/RETROSPECTIVE_PROSPECTIVE_KCL_SSNAP_LESIONS/' + indexing_file)
            ssnap_indices = indexing['Unnamed: 0']

            if data_source == 'nativeNI':
            
                ssnap_subset = dataset_NI.iloc[ssnap_indices,:]  
                pp_ssnap_subset = pre_processing_ssnap.agglomerate_ethnic_groups(ssnap_subset)
                pp_ssnap_subset = pre_processing_ssnap.select_variables(pp_ssnap_subset, predictors_stage)
                pp_ssnap_subset = pd.get_dummies(pp_ssnap_subset, columns=pp_ssnap_subset.select_dtypes('object').columns, prefix_sep='__')
                ni_features = pp_ssnap_subset.loc[:, ~pp_ssnap_subset.columns.str.startswith('S7')]
                       
                for i, event in enumerate(events):

                    target = get_targets(dataset_NI, event, ssnap_indices)
                    
                    if event == 'thromboloysis':          
                        reduced_ni_features = ni_features.loc[:, ~ni_features.columns.str.startswith('S2Thrombolysis')]
                    
                    train_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '__' + event + '__' + 'trainData.csv'
                    test_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '__' + event + '__' + 'testData_.csv'
                    train_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '__' + event + '__' + 'trainTargets.csv'
                    test_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '__' + event + '__' + 'testTargets.csv'

                    X_train = ni_features.loc[indexing.index[indexing['retrospectiveFlag'] == True].tolist(), :]
                    X_test = ni_features.loc[indexing.index[indexing['retrospectiveFlag'] == False].tolist(), :]
                     
                    if event == 'thromboloysis': 
                        X_train = reduced_ni_features.loc[indexing.index[indexing['retrospectiveFlag'] == True].tolist(), :]
                        X_test = reduced_ni_features.loc[indexing.index[indexing['retrospectiveFlag'] == False].tolist(), :]

                    y_train = target.loc[indexing.index[indexing['retrospectiveFlag'] == True].tolist()]
                    y_test = target.loc[indexing.index[indexing['retrospectiveFlag'] == False].tolist()]
                        
                    pd.DataFrame(X_train).to_csv(train_data_filename)
                    pd.DataFrame(X_test).to_csv(test_data_filename)
                    pd.DataFrame(y_train).to_csv(train_targets_filename)
                    pd.DataFrame(y_test).to_csv(test_targets_filename)


            if data_source == 'embeddedLesions':

##                train_lesion_file = 'KCH__retrospective__LESION_voxels_fwhm4_resized4mm_th4.csv'
##                test_lesion_file = 'KCH__prospective__LESION_voxels_fwhm4_resized4mm_th1.csv'
                lesion_file = 'KCH__retrospective_and_prospective__LESION_voxels_fwhm4_resized4mm_th5.csv'
                
                ncomponents = 80
##                train_lesion_data = pd.read_csv(datasets_path + 'SEGMENTATIONS/RETROSPECTIVE_PROSPECTIVE_KCL_SSNAP_LESIONS/' + train_lesion_file)
##                test_lesion_data = pd.read_csv(datasets_path + 'SEGMENTATIONS/RETROSPECTIVE_PROSPECTIVE_KCL_SSNAP_LESIONS/' + test_lesion_file)
                lesion_data = pd.read_csv(datasets_path + 'SEGMENTATIONS/RETROSPECTIVE_PROSPECTIVE_KCL_SSNAP_LESIONS/' + lesion_file)
      
##                train_lesion_features = train_lesion_data.drop(['imgKey', 'ssnapIndex'], axis=1)
##                test_lesion_features = test_lesion_data.drop(['imgKey', 'ssnapIndex'], axis=1)
                lesion_features = lesion_data.drop(['imgKey', 'ssnapIndex', 'retrospectiveFlag'], axis=1)

##                X_train = train_lesion_features.loc[indexing.index[indexing['retrospectiveFlag'] == True].tolist(), :]
##                X_test = lesion_features.loc[indexing.index[indexing['retrospectiveFlag'] == False].tolist(), :]
                X_train = lesion_features.loc[indexing.index[indexing['retrospectiveFlag'] == True].tolist(), :]
                X_test = lesion_features.loc[indexing.index[indexing['retrospectiveFlag'] == False].tolist(), :]
                
                nmf_model = NMF(n_components = ncomponents, random_state=0)
                nmf_model.fit(X_train)
                X_train_nmf = nmf_model.transform(X_train)
                X_test_nmf = nmf_model.transform(X_test)
                
                for i, event in enumerate(events):
                    
                    target = get_targets(dataset_NI, event, ssnap_indices)
                    
                    train_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'trainData.csv'
                    test_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'testData_.csv'
                    train_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'trainTargets.csv'
                    test_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'testTargets.csv'              

                    y_train = target.loc[indexing.index[indexing['retrospectiveFlag'] == True].tolist()]
                    y_test = target.loc[indexing.index[indexing['retrospectiveFlag'] == False].tolist()]
                    
                    pd.DataFrame(X_train_nmf).to_csv(train_data_filename)
                    pd.DataFrame(X_test_nmf).to_csv(test_data_filename)
                    pd.DataFrame(y_train).to_csv(train_targets_filename)
                    pd.DataFrame(y_test).to_csv(test_targets_filename)



            if data_source == 'combinedEmbeddedLesionsNI':
                
                lesion_file = 'KCH__retrospective_and_prospective__LESION_voxels_fwhm4_resized4mm_th5.csv'
                ncomponents = 80
                lesion_data = pd.read_csv(datasets_path + 'SEGMENTATIONS/RETROSPECTIVE_PROSPECTIVE_KCL_SSNAP_LESIONS/' + lesion_file)          
      
                lesion_features = lesion_data.drop(['imgKey', 'ssnapIndex', 'retrospectiveFlag'], axis=1)

                ssnap_subset = dataset_NI.iloc[ssnap_indices,:]  
                pp_ssnap_subset = pre_processing_ssnap.agglomerate_ethnic_groups(ssnap_subset)
                pp_ssnap_subset = pre_processing_ssnap.select_variables(pp_ssnap_subset, predictors_stage)
                pp_ssnap_subset = pd.get_dummies(pp_ssnap_subset, columns=pp_ssnap_subset.select_dtypes('object').columns, prefix_sep='__')
                ni_features = pp_ssnap_subset.loc[:, ~pp_ssnap_subset.columns.str.startswith('S7')]
                ni_features = ni_features.reset_index(drop=True)

                NI_train = ni_features.loc[indexing.index[indexing['retrospectiveFlag'] == True].tolist(), :]
                NI_test = ni_features.loc[indexing.index[indexing['retrospectiveFlag'] == False].tolist(), :]

                L_train = lesion_features.loc[indexing.index[indexing['retrospectiveFlag'] == True].tolist(), :]
                L_test = lesion_features.loc[indexing.index[indexing['retrospectiveFlag'] == False].tolist(), :]

                nmf_model = NMF(n_components = ncomponents, random_state=0)
                nmf_model.fit(L_train)
                L_train_nmf = nmf_model.transform(L_train)
                L_test_nmf = nmf_model.transform(L_test)


                for i, event in enumerate(events):
                    
                    target = get_targets(dataset_NI, event, ssnap_indices)
                  
                    train_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'trainData.csv'
                    test_data_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'testData_.csv'
                    train_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'trainTargets.csv'
                    test_targets_filename = datasets_path + stroke_type + '__' + cv_type + '__' + timeframe + '__' + data_source + '_NMF_' + str(ncomponents) + 'C__' + event + '__' + 'testTargets.csv'

                    X_train_nmf = pd.concat([NI_train.reset_index(), pd.DataFrame(L_train_nmf)], axis = 1)
                    X_test_nmf = pd.concat([NI_test.reset_index(), pd.DataFrame(L_test_nmf)], axis = 1)

                    if event == 'thromboloysis':
                        reduced_NI_train = NI_train.loc[:, ~NI_train.columns.str.startswith('S2Thrombolysis')]
                        reduced_NI_test = NI_test.loc[:, ~NI_test.columns.str.startswith('S2Thrombolysis')]
                        X_train_nmf = pd.concat([reduced_NI_train.reset_index(), pd.DataFrame(L_train_nmf)], axis = 1)
                        X_test_nmf = pd.concat([reduced_NI_test.reset_index(), pd.DataFrame(L_test_nmf)], axis = 1)
                                              
                    y_train = target.loc[indexing.index[indexing['retrospectiveFlag'] == True].tolist()]
                    y_test = target.loc[indexing.index[indexing['retrospectiveFlag'] == False].tolist()]
##                    
                    pd.DataFrame(X_train_nmf).to_csv(train_data_filename)
                    pd.DataFrame(X_test_nmf).to_csv(test_data_filename)
                    pd.DataFrame(y_train).to_csv(train_targets_filename)
                    pd.DataFrame(y_test).to_csv(test_targets_filename)
