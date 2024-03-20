"""Stroke Pre-Processor

This module contains methods to pre-process data from SSNAP
    (Sentinel Stroke National Audit Programme). It replaces part
    of missing data based on assumptions or conditional adjustments.

It contains the following functions:

    * prepare_data
    * make_conditional_adjustments
    * make_unconditional_adjustments
    * correct_noise
    * agglomerate_ethnic_groups
    * select_variables
    * select_cases
    * harmonise_types
    * visualise_distributions
    * remove_mismatching_variables
    * harmonise_categories
    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def make_unconditional_adjustments(dt):
    """Adjustments based only on values of the specific variables

    Parameters
    ----------
    dt : Pandas DataFrame
        All selected variables

    Returns
    -------
    Pandas DataFrame
        Dataset with values in specific variables replaced according to their own range
    """

    # Assign NaN to Age < 18
    dt.loc[dt['S1AgeOnArrival'] < 18, 'S1AgeOnArrival'] = np.nan

    # Assign 'N' to empty co-morbidities: (Congestive Heart Failure,
    # hypertension, Atrial Fibrillation, Diabetes, and Stroke TIA)
    dt.loc[dt['S2CoMCongestiveHeartFailure'].isna(), 'S2CoMCongestiveHeartFailure'] = 'N'
    dt.loc[dt['S2CoMHypertension'].isna(), 'S2CoMHypertension'] = 'N'
    dt.loc[dt['S2CoMAtrialFibrillation'].isna(), 'S2CoMAtrialFibrillation'] = 'N'
    dt.loc[dt['S2CoMDiabetes'].isna(), 'S2CoMDiabetes'] = 'N'
    dt.loc[dt['S2CoMStrokeTIA'].isna(), 'S2CoMStrokeTIA'] = 'N'

    # Assign NaN to values -1 in NIHSS components
    dt.loc[dt.S2NihssArrivalLoc == -1, 'S2NihssArrivalLoc'] = np.nan
    dt.loc[dt.S2NihssArrivalLocQuestions == -1, 'S2NihssArrivalLocQuestions'] = np.nan
    dt.loc[dt.S2NihssArrivalLocCommands == -1, 'S2NihssArrivalLocCommands'] = np.nan
    dt.loc[dt.S2NihssArrivalBestGaze == -1, 'S2NihssArrivalBestGaze'] = np.nan
    dt.loc[dt.S2NihssArrivalVisual == -1, 'S2NihssArrivalVisual'] = np.nan
    dt.loc[dt.S2NihssArrivalFacialPalsy == -1, 'S2NihssArrivalFacialPalsy'] = np.nan
    dt.loc[dt.S2NihssArrivalMotorArmLeft == -1, 'S2NihssArrivalMotorArmLeft'] = np.nan
    dt.loc[dt.S2NihssArrivalMotorArmRight == -1, 'S2NihssArrivalMotorArmRight'] = np.nan
    dt.loc[dt.S2NihssArrivalMotorLegLeft == -1, 'S2NihssArrivalMotorLegLeft'] = np.nan
    dt.loc[dt.S2NihssArrivalMotorLegRight == -1, 'S2NihssArrivalMotorLegRight'] = np.nan
    dt.loc[dt.S2NihssArrivalLimbAtaxia == -1, 'S2NihssArrivalLimbAtaxia'] = np.nan
    dt.loc[dt.S2NihssArrivalSensory == -1, 'S2NihssArrivalSensory'] = np.nan
    dt.loc[dt.S2NihssArrivalBestLanguage == -1, 'S2NihssArrivalBestLanguage'] = np.nan
    dt.loc[dt.S2NihssArrivalDysarthria == -1, 'S2NihssArrivalDysarthria'] = np.nan
    dt.loc[dt.S2NihssArrivalExtinctionInattention == -1, 'S2NihssArrivalExtinctionInattention'] = np.nan

    # Assign 'N' to empty values for Thrombolysis
    dt.loc[dt['S2Thrombolysis'].isna(), 'S2Thrombolysis'] = 'N'

    # Assign 'N' to empty values of 'Palliative Care' and 'End of Life Pathway'
    dt.loc[dt['S3EndOfLifePathway'].isna(), 'S3EndOfLifePathway'] = 'N'
    dt.loc[dt['S3PalliativeCare'].isna(), 'S3PalliativeCare'] = 'N'

    # Assign 'N'  to empty values of Physio, OccTher, Psychology and SpeechLang in S4.
    dt.loc[dt['S4Physio'].isna(), 'S4Physio'] = 'N'
    dt.loc[dt['S4OccTher'].isna(), 'S4OccTher'] = 'N'
    dt.loc[dt['S4Psychology'].isna(), 'S4Psychology'] = 'N'
    dt.loc[dt['S4SpeechLang'].isna(), 'S4SpeechLang'] = 'N'

    # Assign 'No' to empty values and to 'KN' in S5UrinaryTractInfection7Days
    # and S5PneumoniaAntibiotics7Days.
    dt.loc[dt['S5UrinaryTractInfection7Days'].isna(), 'S5UrinaryTractInfection7Days'] = 'N'
    dt.loc[dt['S5UrinaryTractInfection7Days'] == 'KN', 'S5UrinaryTractInfection7Days'] = 'N'
    dt.loc[dt['S5PneumoniaAntibiotics7Days'].isna(), 'S5PneumoniaAntibiotics7Days'] = 'N'
    dt.loc[dt['S5PneumoniaAntibiotics7Days'] == 'KN', 'S5PneumoniaAntibiotics7Days'] = 'N'

    # Assign 'NS' (not screened) to empty 'S6MalnutritionScreening'
    dt.loc[dt['S6MalnutritionScreening'].isna(), 'S6MalnutritionScreening'] = 'NS'

    # Assign 'N' to empty values of 'Palliative Care by discharge'
    # and 'End of Life Pathway by discharge' in S6
    dt.loc[dt['S6PalliativeCareByDischarge'].isna(), 'S6PalliativeCareByDischarge'] = 'N'
    dt.loc[dt['S6EndOfLifePathwayByDischarge'].isna(), 'S6EndOfLifePathwayByDischarge'] = 'N'

    # Assign 'No' to empty 'S6IntPneumaticComp'
    dt.loc[dt['S6IntPneumaticComp'].isna(), 'S6IntPneumaticComp'] = 'N'

    return dt


def make_conditional_adjustments(dt):
    """Adjustments conditioned to other variables

    Parameters
    ----------
    dt : Pandas DataFrame
        All selected variables

    Returns
    -------
    Pandas DataFrame
        Dataset with values in specific variables replaced according to other
        variables
    """

    # Assign 'N' to S1ArriveByAmbulance for patients with onset in hospital
    dt.loc[dt['S1OnsetInHospital'] == 'Y', 'S1ArriveByAmbulance'] = 'N'

    # AF antiplatelet
    dt.loc[dt['S2CoMAtrialFibrillation'] == 'N', 'S2CoMAFAntiplatelet'] = 'NAp'

    # AF anticoagulent
    dt.loc[dt['S2CoMAtrialFibrillation'] == 'N', 'S2CoMAFAnticoagulent'] = 'NAp'

    # Assign 'Not applicable'to 'SwallowScreening4HrsNotPerformedReason' for
    # patients screened for swallow at 4hrs.
    dt.loc[dt['S2SwallowScreening4HrsNotPerformed'] == False, 'S2SwallowScreening4HrsNotPerformedReason'] = 'NAp'

    # Assign 'Not applicable' to 'SwallowScreening72rsNotPerformedReason' for
    # patients screened for swallow at 72hrs or with status 'Not known'.
    # Same for S3OccTherapist72Hrs, S3Physio72Hrs, S3SpLangTherapistComm72Hrs,
    # and 3SpLangTherapistSwallow72Hrs
    dt.loc[dt['S3SwallowScreening72HrsNotPerformed'] == False, 'S3SwallowScreening72HrsNotPerformedReason'] = 'NAp'
    dt.loc[dt['S3SwallowScreening72HrsNotPerformedReason'].isna(), 'S3SwallowScreening72HrsNotPerformedReason'] = 'NK'

    dt.loc[dt['S3OccTherapist72HrsNotAssessed'] == False, 'S3OccTherapist72HrsNotAssessedReason'] = 'NAp'
    dt.loc[dt['S3OccTherapist72HrsNotAssessedReason'].isna(), 'S3OccTherapist72HrsNotAssessedReason'] = 'NK'

    dt.loc[dt['S3Physio72HrsNotAssessed'] == False, 'S3Physio72HrsNotAssessedReason'] = 'NAp'
    dt.loc[dt['S3Physio72HrsNotAssessedReason'].isna(), 'S3Physio72HrsNotAssessedReason'] = 'NK'

    dt.loc[dt['S3SpLangTherapistComm72HrsNotAssessed'] == False, 'S3SpLangTherapistComm72HrsNotAssessedReason'] = 'NAp'
    dt.loc[dt['S3SpLangTherapistComm72HrsNotAssessedReason'].isna(), 'S3PalliativeCare'] = 'NK'

    dt.loc[dt[
               'S3SpLangTherapistSwallow72HrsNotAssessed'] == False, 'S3SpLangTherapistSwallow72HrsNotAssessedReason'] = 'NAp'
    dt.loc[dt['S3SpLangTherapistSwallow72HrsNotAssessedReason'].isna(), 'S3PalliativeCare'] = 'NK'

    # S1 First Ward' assigned to empty 'S4 First Ward

    # Assign 'Not applicable'  to 'S4RehabGoalsNoneReason' for patients who have
    # rehab goals (S4RehabGoalsNone == 0). Remaining empty values were assigned
    # 'Not known'.
    dt.loc[dt['S4RehabGoalsNone'] == 0, 'S4RehabGoalsNoneReason'] = 'NAp'
    dt.loc[dt['S4RehabGoalsNoneReason'].isna(), 'S4RehabGoalsNoneReason'] = 'NK'

    # Assign S2NihssArrivalLoc  to empty S5LocWorst7Days.
    dt.loc[dt['S5LocWorst7Days'].isna(), 'S5LocWorst7Days'] = dt.loc[dt['S5LocWorst7Days'].isna(), 'S2NihssArrivalLoc']

    # Assign 'Not applicable'  to 'S6OccTherapistByDischargeNotAssessedReason'
    # for patients that were assessed (S6OccTherapistByDischargeNotAssessed==0).
    # Remaining empty values assigned status 'not known'
    # Same approach for S6PhysioByDischargeNotAssessed,
    # S6SpLangTherapistCommByDischargeNotAssessed,
    # S6SpLangTherapistCommByDischargeNotAssessed, and
    # S6SpLangTherapistSwallowByDischargeNotAssessed
    dt.loc[dt['S6OccTherapistByDischargeNotAssessed'] == 0, 'S6OccTherapistByDischargeNotAssessedReason'] = 'NAp'
    dt.loc[dt['S6OccTherapistByDischargeNotAssessedReason'].isna(), 'S6OccTherapistByDischargeNotAssessedReason'] = 'NK'

    dt.loc[dt['S6PhysioByDischargeNotAssessed'] == 0, 'S6PhysioByDischargeNotAssessedReason'] = 'NAp'
    dt.loc[dt['S6PhysioByDischargeNotAssessedReason'].isna(), 'S6PhysioByDischargeNotAssessedReason'] = 'NK'

    dt.loc[dt[
               'S6SpLangTherapistCommByDischargeNotAssessed'] == 0, 'S6SpLangTherapistCommByDischargeNotAssessedReason'] = 'NAp'
    dt.loc[dt[
        'S6SpLangTherapistCommByDischargeNotAssessedReason'].isna(), 'S6SpLangTherapistCommByDischargeNotAssessedReason'] = 'NK'

    dt.loc[dt[
               'S6SpLangTherapistSwallowByDischargeNotAssessed'] == 0, 'S6SpLangTherapistSwallowByDischargeNotAssessedReason'] = 'NAp'
    dt.loc[dt[
        'S6SpLangTherapistSwallowByDischargeNotAssessedReason'].isna(), 'S6SpLangTherapistSwallowByDischargeNotAssessedReason'] = 'NK'

    # Assign 'Not applicable'  to 'S6UrinaryContinencePlanNoPlanReason'
    # for patients who have a plan (S6UrinaryContinencePlanNoPlan==0).
    # Remaining empty values assigned status 'not known'
    dt.loc[dt['S6UrinaryContinencePlanNoPlan'] == 0, 'S6UrinaryContinencePlanNoPlanReason'] = 'NAp'
    dt.loc[dt['S6UrinaryContinencePlanNoPlanReason'].isna(), 'S6UrinaryContinencePlanNoPlanReason'] = 'NK'

    # Assign 'Not applicable' to 'S6MoodScreeningNoScreeningReason' for
    # patients who were screened (S6MoodScreeningNoScreening==0).
    # Remaining empty values assigned status 'not known'
    # Same approach for S6CognitionScreeningNoScreeningReason.
    dt.loc[dt['S6MoodScreeningNoScreening'] == 0, 'S6MoodScreeningNoScreeningReason'] = 'NAp'
    dt.loc[dt['S6MoodScreeningNoScreeningReason'].isna(), 'S6MoodScreeningNoScreeningReason'] = 'NK'

    dt.loc[dt['S6CognitionScreeningNoScreening'] == 0, 'S6CognitionScreeningNoScreeningReason'] = 'NAp'
    dt.loc[dt['S6CognitionScreeningNoScreeningReason'].isna(), 'S6CognitionScreeningNoScreeningReason'] = 'NK'

    # Assign 'Not applicable' to 'S7StrokeUnitDeath' for patients
    # who did not die in hospital (S7DischargeType not 'D') or
    # who did not stay on stroke unit (S4StrokeUnitArrivalNA==1)
    dt.loc[dt['S7DischargeType'] != 'D', 'S7StrokeUnitDeath'] = 'NAp'
    dt.loc[dt['S4StrokeUnitArrivalNA'] == 1, 'S7StrokeUnitDeath'] = 'NAp'

    # Assign 'Not applicable' to 'S7CareHomeDischarge' (previous resident or not)
    # for patients who were not discharged to a care home (S7DischargeType not 'CH')
    dt.loc[dt['S7DischargeType'] != 'CH', 'S7CareHomeDischarge'] = 'NAp'

    # Assign 'Not applicable' to 'S7HomeDischargeType' (living alone or not)
    # for patients that were not discharged to home (S7DischargeType not 'H')
    dt.loc[dt['S7DischargeType'] != 'H', 'S7HomeDischargeType'] = 'NAp'

    # Assign 'Not applicable' to 'S7DischargedEsdmt', 'S7DischargedMcrt',
    # and 'S7AdlHelp' for patients that were not discharged
    # (i.e. died or were transferred)
    dt.loc[dt['S7DischargeType'] == 'D', 'S7DischargedEsdmt'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'T', 'S7DischargedEsdmt'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'TN', 'S7DischargedEsdmt'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'D', 'S7DischargedMcrt'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'T', 'S7DischargedMcrt'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'TN', 'S7DischargedMcrt'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'D', 'S7AdlHelp'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'T', 'S7AdlHelp'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'TN', 'S7AdlHelp'] = 'NAp'

    # Assign 'Not applicable' to 'S7AdlHelpType' for patients who did not
    # require help with activities of daily living
    dt.loc[dt['S7AdlHelp'] != 'Y', 'S7AdlHelpType'] = 'NAp'

    # Assign 'Not applicable' to 'S7DischargeAtrialFibrillation' for patients
    # who were not discharged (i.e. died or were transferred)
    dt.loc[dt['S7DischargeType'] == 'D', 'S7DischargeAtrialFibrillation'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'T', 'S7DischargeAtrialFibrillation'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'TN', 'S7DischargeAtrialFibrillation'] = 'NAp'

    # Assign 'Not applicable' to 'S7DischargeAtrialFibrillationAnticoagulation'
    # for patients who were not in atrial fibrillation on discharge
    # (S7DischargeAtrialFibrillation~='Y')
    dt.loc[dt['S7DischargeAtrialFibrillation'] != 'Y', 'S7DischargeAtrialFibrillationAnticoagulation'] = 'NAp'

    # Assign 'Not applicable'  to 'S7DischargeJointCarePlanning' for patients
    # who were not discharged (i.e. died or were transferred).
    # Remaining missing values are assigned 'N'
    dt.loc[dt['S7DischargeType'] == 'D', 'S7DischargeJointCarePlanning'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'T', 'S7DischargeJointCarePlanning'] = 'NAp'
    dt.loc[dt['S7DischargeType'] == 'TN', 'S7DischargeJointCarePlanning'] = 'NAp'
    dt.loc[dt['S7DischargeJointCarePlanning'].isna(), 'S7DischargeJointCarePlanning'] = 'N'

    return dt


def correct_noise(dt):
    """Address potential noise observed in the dataset (mostly non-stationarity issues)

    Parameters
    ----------
    dt : Pandas DataFrame
        All selected variables

    Returns
    -------
    Pandas DataFrame
        Dataset with values in specific variables corrected using info from other variables
    """

    # Address non-stationarity in stroke type
    pih_indices_from_type = dt.index[dt['S2StrokeType'] == 'PIH'].to_list()
    pih_indices_from_procedure = dt.index[dt['S2ThrombolysisNoButHaemorrhagic'] == True].to_list()

    s = set(pih_indices_from_type)
    list_diff = [x for x in pih_indices_from_procedure if x not in s]

    dt.loc[list_diff, 'S2StrokeType'] = 'PIH'

    return dt


def prepare_data(dt_filename, vlist):
    """Selects a pre-determined subset of variables, calls functions to adjust
    their content and drop cases according to stroke type and time from onset to hospital.

    Parameters
    ----------
    dt : Pandas Dataframe
        Non-imaging data
    file_ext : str
        File extensin
    stype: str
        Stroke type

    Returns
    -------
    Pandas DataFrame
    """
    split_tup = os.path.splitext(dt_filename)
    file_extension = split_tup[1]

    if file_extension == '.xlsx':
        data = pd.read_excel(dt_filename)

    if file_extension == '.csv':
        data = pd.read_csv(dt_filename, dtype={'S2ThrombolysisComplications': 'str',
                                               'S2ThrombolysisComplicationOtherDetails': 'str'})

    filename_parts = os.path.split(dt_filename)
    print('\nSSNAP Pre-processing module - file ' + filename_parts[1] + '...')

    selected_data = data.loc[:, vlist]

    adj1_dataset = make_unconditional_adjustments(selected_data)
    adj2_dataset = make_conditional_adjustments(adj1_dataset)
    preprocessed_dataset = correct_noise(adj2_dataset)

    # Add time from onset to hospital arrival

##    aux_oh = preprocessed_dataset['S1FirstArrivalDateTime'] - preprocessed_dataset['S1OnsetDateTime']
##
##    if file_extension == '.csv':
##        aux_oh_hours = aux_oh * 24;  # difference in hours
##    if file_extension == '.xlsx':
##        aux_oh_hours = aux_oh / np.timedelta64(1, 'h')
##
##    aux_oh_hours.loc[aux_oh_hours < 0] = 0  # zero assigned to negative deltas (probably patients already in hospital)
##
##    # oht_prct = np.percentile(aux_oh_hours, [1, 99])
##    # aux_oh_hours.loc[aux_oh_hours>oht_prct[1]] = np.nan # Values higher than 99th percentile removed
##    aux_oh_hours.loc[aux_oh_hours > 360] = np.nan  # Threshold in 15 days
##    preprocessed_dataset.loc[:, 'TimeFromOnsetToHospital'] = aux_oh_hours
##
##    # Add time from hospital arrival to stroke unit
##
##    aux_hsu = preprocessed_dataset['S1FirstStrokeUnitArrivalDateTime'] - preprocessed_dataset['S1FirstArrivalDateTime']
##
##    if file_extension == '.csv':
##        aux_hsu_hours = aux_hsu * 24;  # difference in hours
##    if file_extension == '.xlsx':
##        aux_hsu_hours = aux_hsu / np.timedelta64(1, 'h')
##
##    aux_hsu_hours.loc[aux_hsu_hours < 0] = 0
##    # hsut_prct = np.nanpercentile(aux_hsu_hours, [1, 99])
##    # aux_hsu_hours.loc[aux_hsu_hours > hsut_prct[1]] = np.nan  # Values higher than 99th percentile removed
##    aux_hsu_hours.loc[aux_hsu_hours > 360] = np.nan  # Threshold in 15 days
##    preprocessed_dataset.loc[:, 'TimeFromHospitalArrivalToSU'] = aux_hsu_hours
##
##    # Remove variables - original dates
##    preprocessed_dataset.drop(['S1FirstArrivalDateTime', 'S1OnsetDateTime',
##                               'S1FirstStrokeUnitArrivalDateTime'], axis=1, inplace=True)

    return preprocessed_dataset



def agglomerate_ethnic_groups(dt):
    """Agglomerate ethnic values into 5 groups

        Parameters
        ----------
        dt : Pandas DataFrame
            Pre-processed SSNAP data

        Returns
        -------
        Pandas DataFrame
            Dataset with Ethnicity replaced by ethnic groups.
    """

    return dt.replace({'S1Ethnicity': {'A': 'G1', 'B': 'G1', 'C': 'G1',
                                       'D': 'G2', 'E': 'G2', 'F': 'G2', 'G': 'G2',
                                       'H': 'G3', 'J': 'G3', 'K': 'G3', 'L': 'G3',
                                       'M': 'G4', 'N': 'G4', 'P': 'G4',
                                       'R': 'G5', 'S': 'G5', 'Z': 'G5', '99': 'G5', 99: 'G5'}})


def select_variables(dt, stage):
    """Remove variables according to stroke stage.

        Parameters
        ----------
        dt : Pandas DataFrame
            Pre-processed SSNAP data
        stage: str
            Stroke stage ('72 hours' or 'anytime').

        Returns
        -------
        Pandas DataFrame
            Pre-processed SSNAP data variables excluded according to stroke stage.
    """

    if stage == '48hours':
        varset = dt.loc[:, ~dt.columns.str.startswith('S6')]  # Remove variables from section 6
        # (Assessments – By discharge)
        varset = varset.loc[:,~varset.columns.str.startswith('S5')] # Remove variables from section 5
        # (Patient Condition in first 7 days)
        varset = varset.loc[:, ~varset.columns.str.startswith('S4')]  # Remove variables from section 4
        # ('This admission')
        varset = varset.loc[:, ~varset.columns.str.startswith('S3')]  # Remove variables from section 3
        # ('Assessments – First 72 hours')
        return varset
    else:
        return dt


def select_cases(dt, type):
    """Remove cases according to stroke type and stage.

        Parameters
        ----------
        dt : Pandas DataFrame
            Pre-processed SSNAP data
        type: str
            Stroke type: 'ischaemic', 'haemorrhagic', or 'all'
        stage: str
            Stroke stage ('72 hours' or 'anytime').

        Returns
        -------
        Pandas DataFrame
            Pre-processed SSNAP data with cases excluded according to stroke type and stage.
    """
    if type == 'ischaemic':
        type_ind = np.where(dt['S2StrokeType'] != 'I')[0]
    elif type == 'haemorrhagic':
        type_ind = np.where(dt['S2StrokeType'] != 'PIH')[0]
    else:
        type_ind = np.array([])
    missing_ind = np.where(dt.isna())[0]
    drop_indices = np.unique(np.concatenate((type_ind, missing_ind), 0))

    return dt.drop(drop_indices)


def harmonise_types(dt1, dt2):
    """Harmonise data types across datasets.

    Parameters
    ----------
    dt1, dt2 : Pandas DataFrame
        Pre-processed SSNAP datasets

    Returns
    -------
    dt1, dt2 : Pandas DataFrame
        Pre-processed SSNAP datasets with types given by first dataset and columns containing scores converted to int.
    """

    dt1_var_names = list(dt1.columns)
    for idx, vname in enumerate(dt1_var_names):
        if vname in dt2.columns:
            dt2[vname] = dt2[vname].astype(dt1[vname].dtype)
            if 'Nihss' in vname or 'Rankin' in vname:
                dt1[vname] = dt1[vname].astype('int')
                dt2[vname] = dt2[vname].astype('int')
    return dt1, dt2


def visualise_distributions(dt1, dt2, dtp, path):
    """Plot distributions of variables (side by side across datasets).

        Parameters
        ----------
        dt1, dt2 : Pandas DataFrame
            Pre-processed SSNAP datasets
        dtp: str
            Data type
        path: str
            Absolute path of directory where figures are saved.
    """

    dtype_selection = dt1.select_dtypes(include=dtp)
    dtype_features = list(dtype_selection.columns)
    nrows = len(dtype_features)
    ncols = 2
    sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 32)})
    plt.figure()
    cont = 0
    for idx, feat in enumerate(dtype_features):
        variable = dt1[feat]
        cont = cont + 1
        plt.subplot(nrows, ncols, cont)
        plt.title(feat)
        if dtp == 'float64':
            plt.hist(variable.to_numpy())
        if dtp == 'bool':
            x = np.array(['False', 'True'])
            aux = variable.value_counts()
            y = np.array([aux[0], aux[1]])
            plt.bar(x, y)
        if dtp == 'object' or dtp == 'int64' or dtp == 'uint8':
            dt1[feat].value_counts().plot(kind='bar', xlabel=feat, ylabel='Count', rot=0)
        cont = cont + 1
        if feat in dt2.columns:
            variable = dt2[feat]
            plt.subplot(nrows, ncols, cont)
            plt.title(feat)
            if dtp == 'float64':
                plt.hist(variable.to_numpy())
            if dtp == 'bool':
                x = np.array(['False', 'True'])
                aux = variable.value_counts()
                y = np.array([aux[0], aux[1]])
                plt.bar(x, y)
            if dtp == 'object' or dtp == 'int64' or dtp == 'uint8':
                dt2[feat].value_counts().plot(kind='bar', xlabel=feat, ylabel='Count', rot=0)
    plt.tight_layout(pad=1.0)
    plt.savefig(path + dtp + '_distribution_plots' + '.png')


def remove_mismatching_variables(dt1, dt2):
    """Remove variables that are not in both datasets

        Parameters
        ----------
        dt1, dt2 : Pandas DataFrame
            Pre-processed SSNAP datasets

        Returns
        -------
        dt1, dt2 : Pandas DataFrames
            Datasets with mismatching variables removed.
    """

    dt1_only = np.setdiff1d(dt1.columns.to_list(), dt2.columns.to_list())
    dt2_only = np.setdiff1d(dt2.columns.to_list(), dt1.columns.to_list())
    print('variables - mismatch 1')
    print(dt1_only)
    print('variables - mismatch 2')
    print(dt2_only)
    for var in dt1_only:
        dt1.drop(var, axis=1)
    for var in dt2_only:
        dt2.drop(var, axis=1)
    return dt1, dt2


def harmonise_categories(dt1, dt2):
    """Include columns (filled with zeroes) for categories missing in one of the datasets (based on the other).

        Parameters
        ----------
        dt1, dt2 : Pandas DataFrame
            Pre-processed SSNAP datasets

        Returns
        -------
        dt1, dt2 : Pandas DataFrames
            Datasets with inclusion of zeroes-filled columns for missing categories.
    """

    dt1_only = np.setdiff1d(dt1.columns.to_list(), dt2.columns.to_list())
    dt2_only = np.setdiff1d(dt2.columns.to_list(), dt1.columns.to_list())
    print('categories - mismatch 1')
    print(dt1_only)
    print('categories - mismatch 2')
    print(dt2_only)
    for var in dt1_only:
       # position = dt1.columns.get_loc(var)
       # nullcolumn = pd.DataFrame(np.zeros(len(dt2.index), dtype=int))
       # nullcolumn = nullcolumn.set_index(dt2.index)
       # dt2.insert(position, var, nullcolumn)
        part1, part2 = var.split('__')
        dt1 = dt1.loc[:, ~dt1.columns.str.startswith(part1)]
    for var in dt2_only:
       # position = dt2.columns.get_loc(var)
       # nullcolumn = pd.DataFrame(np.zeros(len(dt1.index), dtype=int))
       # nullcolumn = nullcolumn.set_index(dt1.index)
       # dt1.insert(position, var, nullcolumn)
        part1, part2 = var.split('__')
        dt2 = dt2.loc[:, ~dt2.columns.str.startswith(part1)]
    dt2 = dt2.reindex(dt1.columns, axis=1)
    return dt1, dt2
