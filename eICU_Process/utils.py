from __future__ import absolute_import
from __future__ import print_function

import os
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
import pandas as pd
import os
from itertools import islice
import argparse
import json
import shutil

from tqdm import tqdm


#Filter on useful column for this benchmark

#Select unique patient id
def cohort_stay_id(patients):
    cohort = patients.patientunitstayid.unique()
    return cohort
#filter on adult patients
#filter those having just one stay in unit
def filter_one_unit_stay(patients):
    cohort_count = patients.groupby(by='uniquepid').count()
    index_cohort = cohort_count[cohort_count['patientunitstayid'] == 1].index
    patients = patients[patients['uniquepid'].isin(index_cohort)]
    return patients

#Filter on useful columns from patient table
def filter_patients_on_columns(patients):
    columns = ['patientunitstayid', 'gender', 'age', 'ethnicity', 'unittype','hospitalid',
               'hospitaldischargeyear', 'hospitaldischargeoffset','unitadmittime24',
               'admissionheight', 'hospitaladmitoffset', 'admissionweight',
               'hospitaldischargestatus', 'unitdischargeoffset', 'unitdischargestatus']
    return patients[columns]

def isExist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"文件夹 '{directory}' 创建成功")

def dataframe_from_csv(path, header=0, index_col=False):
    return pd.read_csv(path, header=header, index_col=index_col)

def filter_patients_on_columns_model(patients):
    columns = ['patientunitstayid','hospitalid','unittype','hour','gender', 'age', 'ethnicity',
               'admissionheight', 'admissionweight','unitdischargeoffset']
    return patients[columns]

#Select unique patient id
def cohort_stay_id(patients):
    cohort = patients.patientunitstayid.unique()
    return cohort
def extract_hour_from_timestamp(df, timestamp_column):
    # Add a new column 'hour' by extracting the hour from the specified timestamp column
    df['hour'] = pd.to_datetime(df[timestamp_column], format='%H:%M:%S').dt.hour
    return df


#Extract data from patient table
def read_patients_table(eicu_path):
    print('==>Get data from the patient table...')
    pats = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv'), index_col=False)
    pats = filter_one_unit_stay(pats)#一次住院
    pats = filter_patients_on_columns(pats)#找列
    pats = extract_hour_from_timestamp(pats, 'unitadmittime24')
    pats = filter_patients_on_columns_model(pats)
    return pats

def read_labels_table(eicu_path):
    print('==>Get data from the apachePatientResult table...')
    labels = dataframe_from_csv(os.path.join(eicu_path, 'apachePatientResult.csv'), index_col=False)
    labels = labels[labels['actualiculos']>=2]
    labels = filter_label_on_columns(labels)
    labels = labels.dropna()
    return labels


def filter_label_on_columns(labels):
    # 选择需要保留的列
    columns_to_keep = ['patientunitstayid', 'actualiculos', 'actualicumortality']

    # 保留需要的列，并去除重复的患者（保留第一次出现的）,两次标签相同
    filtered_labels = labels[columns_to_keep].drop_duplicates(subset='patientunitstayid', keep='first')

    return filtered_labels

## Here we deal with lab table
#Select the useful columns from lab table
def filter_lab_on_columns(lab):
    columns = ['patientunitstayid', 'labresultoffset', 'labname', 'labresult']
    return lab[columns]

#Rename the columns in order to have a unified name
def rename_lab_columns(lab):
    lab.rename(index=str, columns={"labresultoffset": "itemoffset",
                                   "labname": "itemname", "labresult": "itemvalue"}, inplace=True)
    return lab

#Select the lab measurement from lab table
def item_name_selected_from_lab(lab, items):
    lab = lab[lab['itemname'].isin(items)]
    return lab

#Check if the lab measurement is valid
def check(x):
    try:
        x = float(str(x).strip())
    except:
        x = np.nan
    return x
def check_itemvalue(df):
    df['itemvalue'] = df['itemvalue'].apply(lambda x: check(x))
    df['itemvalue'] = df['itemvalue'].astype(float)
    return df

def getLimitTimedata(pats, labs, ncs):
    print('==> Get TimeseriesData in icu stay ...')

    # Step 1: 将 labs 和 ncs 按照 'patientunitstayid' 和 'itemoffset' 排序
    labs = labs.sort_values(by=['patientunitstayid', 'itemoffset'])
    ncs = ncs.sort_values(by=['patientunitstayid', 'itemoffset'])

    # Step 2: 获取每个患者的 unitdischargeoffset，并将其应用于 labs 和 ncs
    pats_with_stop_offset = pats.set_index('patientunitstayid')['unitdischargeoffset']
    labs['stop_offset'] = labs['patientunitstayid'].map(pats_with_stop_offset)
    ncs['stop_offset'] = ncs['patientunitstayid'].map(pats_with_stop_offset)

    # Step 3: 边界时间点赋值
    def nearest_to_stop_offset(df):
        nearest_0_idx = (np.abs(df['itemoffset'])).idxmin()
        nearest_stop_idx = (np.abs(df['itemoffset'] - df['stop_offset'])).idxmin()

        nearest_0_row = df.loc[nearest_0_idx].copy()
        nearest_0_row['itemoffset'] = 0

        nearest_stop_row = df.loc[nearest_stop_idx].copy()
        nearest_stop_row['itemoffset'] = nearest_stop_row['stop_offset']

        df = pd.concat([df, nearest_0_row, nearest_stop_row], axis=0)

        return df
    tqdm.pandas(desc="Processing labs")
    labs = labs.groupby(['patientunitstayid', 'itemname']).progress_apply(nearest_to_stop_offset)

    tqdm.pandas(desc="Processing ncs")
    ncs = ncs.groupby(['patientunitstayid', 'itemname']).progress_apply(nearest_to_stop_offset)

    # Step 4: 删除所有'itemoffset'小于0和大于'stop_offset'的记录
    labs = labs[(labs['itemoffset'] >= 0) & (labs['itemoffset'] <= labs['stop_offset'])]
    ncs = ncs[(ncs['itemoffset'] >= 0) & (ncs['itemoffset'] <= ncs['stop_offset'])]

    # Step 5:删除不必要的列
    pats = pats.drop('unitdischargeoffset', axis=1)
    labs = labs[['patientunitstayid','itemoffset','itemname','itemvalue']]
    ncs = ncs[['patientunitstayid','itemoffset','itemname','itemvalue']]

    labs['itemoffset'] = labs['itemoffset'].astype(int)
    labs['patientunitstayid'] = labs['patientunitstayid'].astype(int)
    ncs['itemoffset'] = ncs['itemoffset'].astype(int)
    ncs['patientunitstayid'] = ncs['patientunitstayid'].astype(int)
    labs.dropna()
    ncs.dropna()
    return pats, labs, ncs


#extract the lab items for each patient
def read_lab_table(eicu_path):
    print('==>Get data from the lab table...')
    lab = dataframe_from_csv(os.path.join(eicu_path, 'lab.csv'), index_col=False)
    items = ['bedside glucose', 'glucose', 'pH', 'FiO2']
    lab = filter_lab_on_columns(lab)
    lab = rename_lab_columns(lab)
    lab = item_name_selected_from_lab(lab, items)
    lab.loc[lab['itemname'] == 'bedside glucose', 'itemname'] = 'glucose'  # unify bedside glucose and glucose
    lab = check_itemvalue(lab)
    return lab

#Select the nurseCharting items and save it into nc
#Filter the useful columns from nc table
def filter_nc_on_columns(nc):
    columns = ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevallabel',
               'nursingchartcelltypevalname', 'nursingchartvalue']
    return nc[columns]

#Unify the column names in order to be used later
def rename_nc_columns(nc):
    nc.rename(index=str, columns={"nursingchartoffset": "itemoffset",
                                  "nursingchartcelltypevalname": "itemname",
                                  "nursingchartcelltypevallabel": "itemlabel",
                                  "nursingchartvalue": "itemvalue"}, inplace=True)
    return nc

#Select the items using name and label
def item_name_selected_from_nc(nc, label, name):
    nc = nc[(nc.itemname.isin(name)) |
            (nc.itemlabel.isin(label))]
    return nc

#Convert fahrenheit to celsius
def conv_far_cel(nc):
    nc['itemvalue'] = nc['itemvalue'].astype(float)
    nc.loc[nc['itemname'] == "Temperature (F)", "itemvalue"] = ((nc['itemvalue'] - 32) * (5 / 9))

    return nc

#Unify the different names into one for each measurement
def replace_itemname_value(nc):
    nc.loc[nc['itemname'] == 'Value', 'itemname'] = nc.itemlabel
    nc.loc[nc['itemname'] == 'Non-Invasive BP Systolic', 'itemname'] = 'Invasive BP Systolic'
    nc.loc[nc['itemname'] == 'Non-Invasive BP Diastolic', 'itemname'] = 'Invasive BP Diastolic'
    nc.loc[nc['itemname'] == 'Temperature (F)', 'itemname'] = 'Temperature (C)'
    nc.loc[nc['itemlabel'] == 'Arterial Line MAP (mmHg)', 'itemname'] = 'MAP (mmHg)'
    return nc

def read_nc_table(eicu_path):
    print('==>Get data from the nurseCharting table...')
    # import pdb;pdb.set_trace()
    nc = dataframe_from_csv(os.path.join(eicu_path, 'nurseCharting.csv'), index_col=False)
    nc = filter_nc_on_columns(nc)
    nc = rename_nc_columns(nc)
    typevallabel = ['Glasgow coma score', 'Heart Rate', 'O2 Saturation', 'Respiratory Rate', 'MAP (mmHg)',
                    'Arterial Line MAP (mmHg)']
    typevalname = ['Non-Invasive BP Systolic', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic',
                   'Invasive BP Diastolic', 'Temperature (C)', 'Temperature (F)']
    nc = item_name_selected_from_nc(nc, typevallabel, typevalname)
    nc = check_itemvalue(nc)
    nc = conv_far_cel(nc)
    nc = replace_itemname_value(nc)

    del nc['itemlabel']
    return nc
def read_diagnosis_table(eicu_path):
    print('==>Get data from the diagnosis table...')
    diags = dataframe_from_csv(os.path.join(eicu_path, 'diagnosis.csv'), index_col=False)
    diags = filter_diag_on_columns(diags)
    diags = diags.dropna()
    return diags


def filter_diag_on_columns(diags):
    # 选择需要保留的列
    columns_to_keep = ['patientunitstayid', 'diagnosisstring']
    filtered_diags = diags[columns_to_keep]
    return filtered_diags

#split train&test
def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path
def shuffle_stays(stays, seed=9):
    return shuffle(stays, random_state=seed)

def process_table(table_name, table, stays, folder_path):
    table = table.loc[stays].copy()
    table.to_csv('{}/{}.csv'.format(folder_path, table_name), index=True)





#diag_process
# looks hacky but works and is quite fast
def add_codes(splits, codes_dict, words_dict, count):
    codes = list()
    levels = len(splits)  # the max number of levels is 6
    if levels >= 1:
        try:
            codes.append(codes_dict[splits[0]][0])
            codes_dict[splits[0]][2] += 1
        except KeyError:
            codes_dict[splits[0]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0]
            count += 1
    if levels >= 2:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][0])
            codes_dict[splits[0]][1][splits[1]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1]
            count += 1
    if levels >= 3:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2]
            count += 1
    if levels >= 4:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2] + '|' + splits[3]
            count += 1
    if levels >= 5:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2] + '|' + splits[3] + '|' + splits[4]
            count += 1
    if levels == 6:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][1][splits[5]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][1][splits[5]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][1][splits[5]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2] + '|' + splits[3] + '|' + splits[4] + '|' + splits[5]
            count += 1
    return codes, count


def get_mapping_dict(unique_diagnoses):

    # a lot of the notes strings look the same, so we will not propagate beyond Organ Systems for this:
    main_diagnoses = [a for a in unique_diagnoses if not (a.startswith('notes') or a.startswith('admission'))]
    adm_diagnoses = [a for a in unique_diagnoses if a.startswith('admission diagnosis')]
    pasthistory_organsystems = [a for a in unique_diagnoses if a.startswith('notes/Progress Notes/Past History/Organ Systems/')]
    pasthistory_comments = [a for a in unique_diagnoses if a.startswith('notes/Progress Notes/Past History/Past History Obtain Options')]

    # sort into alphabetical order to keep the codes roughly together numerically.
    main_diagnoses.sort()
    adm_diagnoses.sort()
    pasthistory_organsystems.sort()
    pasthistory_comments.sort()

    mapping_dict = {}
    codes_dict = {}
    words_dict = {}
    count = 0

    for diagnosis in main_diagnoses:
        splits = diagnosis.split('|')
        codes, count = add_codes(splits, codes_dict, words_dict, count)
        # add all codes relevant to the diagnosisstring
        mapping_dict[diagnosis] = codes

    for diagnosis in adm_diagnoses:
        # take out the things that are common to all of these because it creates unnecessary levels
        shortened = diagnosis.replace('admission diagnosis|', '')
        shortened = shortened.replace('All Diagnosis|', '')
        shortened = shortened.replace('Additional APACHE  Information|', '')
        splits = shortened.split('|')
        codes, count = add_codes(splits, codes_dict, words_dict, count)
        mapping_dict[diagnosis] = codes

    for diagnosis in pasthistory_organsystems:
        # take out the things that are common to all of these because it creates unnecessary levels
        shortened = diagnosis.replace('notes/Progress Notes/Past History/Organ Systems/', '')
        splits = shortened.split('/')  # note different split to main_diagnoses
        codes, count = add_codes(splits, codes_dict, words_dict, count)
        # add all codes relevant to the diagnosisstring
        mapping_dict[diagnosis] = codes

    for diagnosis in pasthistory_comments:
        # take out the things that are common to all of these because it creates unnecessary levels
        shortened = diagnosis.replace('notes/Progress Notes/Past History/Past History Obtain Options/', '')
        splits = shortened.split('/')  # note different split to main_diagnoses
        codes, count = add_codes(splits, codes_dict, words_dict, count)
        # add all codes relevant to the diagnosisstring
        mapping_dict[diagnosis] = codes

    return codes_dict, mapping_dict, count, words_dict

# get rid of anything that is a parent to only one child (index 2 is 1)
def find_pointless_codes(diag_dict):
    pointless_codes = []
    for key, value in diag_dict.items():
        # if there is only one child, then the branch is linear and can be condensed
        if value[2] == 1:
            pointless_codes.append(value[0])
        # get rid of repeat copies where the parent and child are the same title
        for next_key, next_value in value[1].items():
            if key.lower() == next_key.lower():
                pointless_codes.append(next_value[0])
        pointless_codes += find_pointless_codes(value[1])
    return pointless_codes

# get rid of any codes that have a frequency of less than cut_off
def find_rare_codes(cut_off, sparse_df):
    prevalence = sparse_df.sum(axis=0)  # see if you can stop it making pointless extra classes
    rare_codes = prevalence.loc[prevalence <= cut_off].index
    return list(rare_codes)

def add_apache_diag(sparse_df, eICU_path, cut_off):

    print('==> Adding admission diagnoses from flat_features.csv...')
    flat = pd.read_csv(eICU_path + 'flat_features.csv')
    adm_diag = flat[['patientunitstayid', 'apacheadmissiondx']]
    adm_diag.set_index('patientunitstayid', inplace=True)
    adm_diag = pd.get_dummies(adm_diag, columns=['apacheadmissiondx'])
    rare_adm_diag = find_rare_codes(cut_off, adm_diag)
    # it could be worth doing some further grouping on the rare_adm_diagnoses before we throw them away
    # the first word is a good approximation
    groupby_dict = {}
    for diag in adm_diag.columns:
        if diag in rare_adm_diag:
            groupby_dict[diag] = 'groupedapacheadmissiondx_' + diag.split(' ', 1)[0].split('/', 1)[0].split(',', 1)[0][18:]
        else:
            groupby_dict[diag] = diag
    adm_diag = adm_diag.groupby(groupby_dict, axis=1).sum()
    rare_adm_diag = find_rare_codes(cut_off, adm_diag)
    adm_diag.drop(columns=rare_adm_diag, inplace=True)
    all_diag = sparse_df.join(adm_diag, how='outer', on='patientunitstayid')
    return all_diag

def diagnoses_main(eICU_path, cut_off_prevalence):

    print('==> Loading data diagnoses.csv...')
    diagnoses = pd.read_csv(eICU_path + 'all_diags.csv')
    diagnoses.set_index('patientunitstayid', inplace=True)

    unique_diagnoses = diagnoses.diagnosisstring.unique()
    codes_dict, mapping_dict, count, words_dict = get_mapping_dict(unique_diagnoses)

    patients = diagnoses.index.unique()
    index_to_patients = dict(enumerate(patients))
    patients_to_index = {v: k for k, v in index_to_patients.items()}

    # reconfiguring the diagnosis data into a dictionary
    diagnoses = diagnoses.groupby('patientunitstayid').apply(lambda diag: diag.to_dict(orient='list')['diagnosisstring']).to_dict()
    diagnoses = {patient: [code for diag in list_diag for code in mapping_dict[diag]] for (patient, list_diag) in diagnoses.items()}

    num_patients = len(patients)
    sparse_diagnoses = np.zeros(shape=(num_patients, count))
    for patient, codes in diagnoses.items():
        sparse_diagnoses[patients_to_index[patient], codes] = 1  # N.B. it doesn't matter that codes contains repeats here

    pointless_codes = find_pointless_codes(codes_dict)

    sparse_df = pd.DataFrame(sparse_diagnoses, index=patients, columns=range(count))
    cut_off = round(cut_off_prevalence*num_patients)
    rare_codes = find_rare_codes(cut_off, sparse_df)
    sparse_df.drop(columns=rare_codes + pointless_codes, inplace=True)
    sparse_df.rename(columns=words_dict, inplace=True)
    #sparse_df = add_apache_diag(sparse_df, eICU_path, cut_off)
    print('==> Keeping ' + str(sparse_df.shape[1]) + ' diagnoses which have a prevalence of more than ' + str(cut_off_prevalence*100) + '%...')

    # filter out any patients that don't have timeseries
    try:
        with open(eICU_path + 'stays.txt', 'r') as f:
            ts_patients = [int(patient.rstrip()) for patient in f.readlines()]
    except FileNotFoundError:
        ts_patients = pd.read_csv(eICU_path + 'preprocessed_timeseries.csv')
        ts_patients = [x for x in ts_patients.patient.unique()]
        with open(eICU_path + 'stays.txt', 'w') as f:
            for patient in ts_patients:
                f.write("%s\n" % patient)
    sparse_df = sparse_df.loc[ts_patients]

    # make naming consistent with the other tables
    sparse_df.rename_axis('patient', inplace=True)
    sparse_df.sort_index(inplace=True)
    sparse_df.fillna(0, inplace=True)  # make sure all values are filled in

    print('==> Saving finalised preprocessed diagnoses...')
    sparse_df.to_csv(eICU_path + 'preprocessed_diagnoses.csv')

    return
#time_process
def reconfigure_timeseries(timeseries, offset_column, feature_column=None, test=False):
    if test:
        timeseries = timeseries.iloc[300000:5000000]
    timeseries.set_index(['patientunitstayid', pd.to_timedelta(timeseries[offset_column], unit='T')], inplace=True)
    timeseries.drop(columns=offset_column, inplace=True)
    if feature_column is not None:
        timeseries = timeseries.pivot_table(columns=feature_column, index=timeseries.index)
    # convert index to multi-index with both patients and timedelta stamp
    timeseries.index = pd.MultiIndex.from_tuples(timeseries.index, names=['patient', 'time'])
    return timeseries

def resample_and_mask(timeseries, eICU_path, header, mask_decay=True, decay_rate=4/3, test=False,
                       verbose=False, length_limit=24*14):
    if test:
        mask_decay = False
        verbose = True
    if verbose:
        print('Resampling to 1 hour intervals...')
    # take the mean of any duplicate index entries for unstacking
    timeseries = timeseries.groupby(level=[0, 1]).mean()

    timeseries.reset_index(level=1, inplace=True)
    timeseries.time = timeseries.time.dt.ceil(freq='H')
    timeseries.set_index('time', append=True, inplace=True)
    timeseries.reset_index(level=0, inplace=True)
    resampled = timeseries.groupby('patient').resample('H', closed='right', label='right').mean().drop(columns='patient')
    del (timeseries)

    def apply_mask_decay(mask_bool):
        mask = mask_bool.astype(int)
        mask.replace({0: np.nan}, inplace=True)  # so that forward fill works
        inv_mask_bool = ~mask_bool
        count_non_measurements = inv_mask_bool.cumsum() - \
                                 inv_mask_bool.cumsum().where(mask_bool).ffill().fillna(0)
        decay_mask = mask.ffill().fillna(0) / (count_non_measurements * decay_rate).replace(0, 1)
        return decay_mask

    # store which values had to be imputed
    if mask_decay:
        if verbose:
            print('Calculating mask decay features...')
        mask_bool = resampled.notnull()
        mask = mask_bool.groupby('patient').transform(apply_mask_decay)
        del (mask_bool)
    else:
        if verbose:
            print('Calculating binary mask features...')
        mask = resampled.notnull()
        mask = mask.astype(int)

    if verbose:
        print('Filling missing data forwards...')
    # carry forward missing values (note they will still be 0 in the nulls table)
    resampled = resampled.ffill()

    # simplify the indexes of both tables
    mask = mask.rename(index=dict(zip(mask.index.levels[1],
                                      mask.index.levels[1].days*24 + mask.index.levels[1].seconds//3600)))
    resampled = resampled.rename(index=dict(zip(resampled.index.levels[1],
                                                resampled.index.levels[1].days*24 +
                                                resampled.index.levels[1].seconds//3600)))

    # clip to length_limit
    if length_limit is not None:
        within_length_limit = resampled.index.get_level_values(1) < length_limit
        resampled = resampled.loc[within_length_limit]
        mask = mask.loc[within_length_limit]

    if verbose:
        print('Filling in remaining values with zeros...')
    resampled.fillna(0, inplace=True)

    # rename the columns in pandas for the mask so it doesn't complain
    mask.columns = [str(col) + '_mask' for col in mask.columns]

    # merge the mask with the features
    final = pd.concat([resampled, mask], axis=1)
    final.reset_index(level=1, inplace=True)
    final = final.loc[final.time > 0]

    if verbose:
        print('Saving progress...')
    # save to csv
    if test is False:
        final.to_csv(eICU_path + 'preprocessed_timeseries.csv', mode='a', header=header)
    return

def gen_patient_chunk(patients, size=1000):
    it = iter(patients)
    chunk = list(islice(it, size))
    while chunk:
        yield chunk
        chunk = list(islice(it, size))

def gen_timeseries_file(eICU_path, test=False):

    print('==> Loading data from timeseries files...')
    if test:
        timeseries_lab = pd.read_csv(eICU_path + 'all_labs.csv', nrows=500000)
        timeseries_nurse = pd.read_csv(eICU_path + 'all_ncs.csv.csv', nrows=500000)

    else:
        timeseries_lab = pd.read_csv(eICU_path + 'all_labs.csv')
        timeseries_nurse = pd.read_csv(eICU_path + 'all_ncs.csv')

    print('==> Reconfiguring lab timeseries...')
    timeseries_lab = reconfigure_timeseries(timeseries_lab,
                                            offset_column='itemoffset',
                                            feature_column='itemname',
                                            test=test)
    timeseries_lab.columns = timeseries_lab.columns.droplevel()



    print('==> Reconfiguring nurse timeseries...')

    timeseries_nurse = reconfigure_timeseries(timeseries_nurse,
                                              offset_column='itemoffset',
                                              feature_column='itemname',
                                              test=test)
    timeseries_nurse.columns = timeseries_nurse.columns.droplevel()
    patients = timeseries_lab.index.unique(level=0)

    size = 4000
    gen_chunks = gen_patient_chunk(patients, size=size)
    i = size
    header = True  # for the first chunk include the header in the csv file

    print('==> Starting main processing loop...')

    for patient_chunk in gen_chunks:
        try:
            merged = pd.concat([timeseries_lab.loc[patient_chunk], timeseries_nurse.loc[patient_chunk]], sort=False)
        except KeyError as e:
            print(str(e))
            invalid_index = int(str(e).split("[")[1].split("]")[0])
            patient_chunk.remove(invalid_index)
            merged = pd.concat([timeseries_lab.loc[patient_chunk], timeseries_nurse.loc[patient_chunk]], sort=False)
        if i == size:  # fixed from first run
            # all if not all are not normally distributed
            quantiles = merged.quantile([0.05, 0.95])
            maxs = quantiles.loc[0.95]
            mins = quantiles.loc[0.05]

        merged = 2 * (merged - mins) / (maxs - mins) - 1

        # we then need to make sure that ridiculous outliers are clipped to something sensible
        merged.clip(lower=-4, upper=4, inplace=True)  # room for +- 3 on each side, as variables are scaled roughly between 0 and 1

        resample_and_mask(merged, eICU_path, header, mask_decay=True, decay_rate=4/3, test=test, verbose=False)
        print('==> Processed ' + str(i) + ' patients...')
        i += size
        header = False

    return

def add_time_of_day(processed_timeseries, flat_features):

    print('==> Adding time of day features...')
    processed_timeseries = processed_timeseries.join(flat_features[['hour']], how='inner', on='patient')
    processed_timeseries['hour'] = processed_timeseries['time'] + processed_timeseries['hour']
    hour_list = np.linspace(0, 1, 24)  # make sure it's still scaled well
    processed_timeseries['hour'] = processed_timeseries['hour'].apply(lambda x: hour_list[x%24 - 24])
    return processed_timeseries

def filter_time_span(df, min_time=48, max_time=200):
    # 找到每个患者的最大时间
    max_times = df.groupby(df.index)['time'].max()
    # 找到时间最大值在给定范围之内的患者
    valid_patients = max_times[(max_times >= min_time) & (max_times <= max_time)].index
    # 保留这些患者的数据
    df_filtered = df[df.index.isin(valid_patients)]
    return df_filtered

def further_processing(eICU_path, test=False):

    if test:
        processed_timeseries = pd.read_csv(eICU_path + 'preprocessed_timeseries.csv', nrows=999999)
    else:
        processed_timeseries = pd.read_csv(eICU_path + 'preprocessed_timeseries.csv')
    processed_timeseries.rename(columns={'Unnamed: 1': 'time'}, inplace=True)
    processed_timeseries.set_index('patient', inplace=True)
    flat_features = pd.read_csv(eICU_path + 'all_flats.csv')
    flat_features.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    processed_timeseries.sort_values(['patient', 'time'], inplace=True)
    flat_features.set_index('patient', inplace=True)

    processed_timeseries = add_time_of_day(processed_timeseries, flat_features)
    #only 15-200 is need
    processed_timeseries = filter_time_span(processed_timeseries)

    if test is False:
        print('==> Saving finalised preprocessed timeseries...')
        # this will replace old one that was updated earlier in the script
        processed_timeseries.to_csv(eICU_path + 'preprocessed_timeseries.csv')

    return

def timeseries_main(eICU_path, test=False):
    # make sure the preprocessed_timeseries.csv file is not there because the first section of this script appends to it
    if test is False:
        print('==> Removing the preprocessed_timeseries.csv file if it exists...')
        try:
            os.remove(eICU_path + 'preprocessed_timeseries.csv')
        except FileNotFoundError:
            pass
    gen_timeseries_file(eICU_path, test)
    further_processing(eICU_path, test)
    return

def preprocess_flat(flat):

    # make naming consistent with the other tables
    flat.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    flat.set_index('patient', inplace=True)


    flat['gender'].replace({'Male': 1, 'Female': 0}, inplace=True)

    cat_features = ['ethnicity']
    # get rid of any really uncommon values
    for f in cat_features:
        too_rare = [value for value, count in flat[f].value_counts().items() if count < 1000]
        flat.loc[flat[f].isin(too_rare), f] = 'misc'

    # convert the categorical features to one-hot
    flat = pd.get_dummies(flat, columns=cat_features, dtype=int)

    # 10 patients have NaN for age; we fill this with the mean value which is 63
    flat['age'].fillna('63', inplace=True)
    # some of the ages are like '> 89' rather than numbers, this needs removing and converting to numbers
    # but we make an extra variable to keep this information
    flat['> 89'] = flat['age'].str.contains('> 89').astype(int)
    flat['age'] = flat['age'].replace('> ', '', regex=True)
    flat['age'] = [float(value) for value in flat.age.values]

    # note that the features imported from the time series have already been normalised
    # standardisation is for features that are probably normally distributed
    features_for_standardisation = 'admissionheight'
    means = flat[features_for_standardisation].mean(axis=0)
    stds = flat[features_for_standardisation].std(axis=0)
    flat[features_for_standardisation] = (flat[features_for_standardisation] - means) / stds

    # probably not normally distributed
    features_for_min_max = ['admissionweight', 'age', 'hour']#, 'bedcount']

    def scale_min_max(flat):
        quantiles = flat.quantile([0.05, 0.95])
        maxs = quantiles.loc[0.95]
        mins = quantiles.loc[0.05]
        return 2 * (flat - mins) / (maxs - mins) - 1

    flat[features_for_min_max] = flat[features_for_min_max].apply(scale_min_max)

    # we then need to make sure that ridiculous outliers are clipped to something sensible
    flat[features_for_standardisation] = flat[features_for_standardisation].clip(lower=-4, upper=4)  # room for +- 3 on each side of the normal range, as variables are scaled roughly between -1 and 1
    flat[features_for_min_max] = flat[features_for_min_max].clip(lower=-4, upper=4)

    # fill in the NaNs
    # these are mainly found in admissionweight and admissionheight,
    # so we create another variable to tell the model when this has been imputed
    flat['nullweight'] = flat['admissionweight'].isnull().astype(int)
    flat['nullheight'] = flat['admissionheight'].isnull().astype(int)
    flat['admissionweight'].fillna(0, inplace=True)
    flat['admissionheight'].fillna(0, inplace=True)
    # there are only 11 missing genders but we might as well set this to 0.5 to tell the model we aren't sure
    flat['gender'].fillna(0.5, inplace=True)
    flat['gender'].replace({'Other': 0.5, 'Unknown': 0.5}, inplace=True)

    return flat

def preprocess_labels(labels):

    # make naming consistent with the other tables
    labels.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    labels.set_index('patient', inplace=True)
    labels['actualicumortality'].replace({'EXPIRED': 1, 'ALIVE': 0}, inplace=True)

    return labels

def flat_and_labels_main(eICU_path):

    print('==> Loading data from labels and flat features files...')
    flat = pd.read_csv(eICU_path + 'all_flats.csv')
    labels = pd.read_csv(eICU_path + 'all_labels.csv')

    hosid = flat['hospitalid']
    labels.insert(1, 'hospitalid', hosid)
    flat = flat.drop('hospitalid', axis=1)

    icuid = flat['unittype']
    labels.insert(1, 'unittype', icuid)
    flat = flat.drop('unittype', axis=1)

    flat = preprocess_flat(flat)
    flat.sort_index(inplace=True)

    labels = preprocess_labels(labels)
    labels.sort_index(inplace=True)

    # filter out any patients that don't have timeseries
    try:
        with open(eICU_path + 'stays.txt', 'r') as f:
            ts_patients = [int(patient.rstrip()) for patient in f.readlines()]
    except FileNotFoundError:
        ts_patients = pd.read_csv(eICU_path + 'preprocessed_timeseries.csv')
        ts_patients = [x for x in ts_patients.patient.unique()]
        with open(eICU_path + 'stays.txt', 'w') as f:
            for patient in ts_patients:
                f.write("%s\n" % patient)
    flat = flat.loc[ts_patients].copy()
    labels = labels.loc[ts_patients].copy()

    print('==> Saving finalised preprocessed labels and flat features...')
    flat.to_csv(eICU_path + 'preprocessed_flat.csv')
    labels.to_csv(eICU_path + 'preprocessed_labels.csv')
    return
#fl_process
def fl_hos_pro(eICU_path,fl_path):
    fl_hos_path = os.path.join(fl_path,'hos')
    isExist(fl_hos_path)
    labels_df = pd.read_csv(os.path.join(eICU_path,'preprocessed_labels.csv'))
    flat_df = pd.read_csv(os.path.join(eICU_path,'preprocessed_flat.csv'))
    timeseries_df = pd.read_csv(os.path.join(eICU_path,'preprocessed_timeseries.csv'))
    diag_df = pd.read_csv(os.path.join(eICU_path,'preprocessed_diagnoses.csv'))
    hospital_patient_dict = labels_df.groupby('hospitalid')['patient'].apply(list).to_dict()
    hos = hospital_patient_dict.keys()
    sort_hos_num = sorted(hospital_patient_dict, key=lambda x: len(hospital_patient_dict[x]), reverse=True)
    with open(os.path.join(fl_hos_path,'hospital_patient_dict.json'), 'w') as file:
        json.dump({key: hospital_patient_dict[key] for key in sort_hos_num}, file)
    for item in hos:
        print(f'在处理中心{item}')
        fl_dir = os.path.join(fl_hos_path, str(item))
        isExist(fl_dir)
        fl_labels_df = labels_df[labels_df['patient'].isin(hospital_patient_dict[item])]
        fl_labels_df = fl_labels_df.drop(['hospitalid'], axis=1)
        fl_labels_df = fl_labels_df.drop(['unittype'], axis=1)
        fl_labels_df.to_csv(os.path.join(fl_dir, 'preprocessed_labels.csv'),index=False)
        fl_flat_df = flat_df[flat_df['patient'].isin(hospital_patient_dict[item])]
        fl_flat_df.to_csv(os.path.join(fl_dir, 'preprocessed_flat.csv'),index=False)
        fl_timeseries_df = timeseries_df[timeseries_df['patient'].isin(hospital_patient_dict[item])]
        fl_timeseries_df.to_csv(os.path.join(fl_dir, 'preprocessed_timeseries.csv'),index=False)
        fl_diag_df = diag_df[diag_df['patient'].isin(hospital_patient_dict[item])]
        fl_diag_df.to_csv(os.path.join(fl_dir, 'preprocessed_diagnoses.csv'),index=False)

def move_choosed_fl_data(fl_path, final_path):
    hos_path = os.path.join(final_path, 'hos')
    isExist(hos_path)

    with open(os.path.join(fl_path, 'hos', 'hospital_patient_dict.json'), 'r') as f:
        all_hos = json.load(f)

    hosid = ['hospital_patient_dict.json']
    for hos in list(all_hos.keys())[0:20]:
        hosid.append(hos)
    for i in hosid:
        original_data_path = os.path.join(fl_path, 'hos', i)
        final_data_path = os.path.join(hos_path, i)
        if os.path.isdir(original_data_path):
            print(f'选择{i}中心，{len(all_hos[i])}人')
            if not os.path.exists(final_data_path):
                shutil.copytree(original_data_path,final_data_path)
        else:
            shutil.copy(original_data_path,final_data_path)

if __name__ == '__main__':
    with open(os.path.join('fl_data', 'hos', 'hospital_patient_dict.json'), 'r') as f:
        all_hos = json.load(f)
    for hos, patient in zip(list(all_hos.keys())[0:20], list(all_hos.values())[0:20]):
        print(f'\'{hos}\':{len(patient)},')


