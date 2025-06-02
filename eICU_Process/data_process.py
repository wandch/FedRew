from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import utils


def filter_common_patients(dataframes,test):
    print('==> find the same id in fine tables...')
    common_patients = set(dataframes[0]['patientunitstayid'].unique())

    for df in dataframes[1:]:
        common_patients &= set(df['patientunitstayid'].unique())
    print(f'==> find {len(common_patients)} common patients...')
    if test:
        common_patients = list(common_patients)[:4000]  # 将集合转换为列表并进行切片操作
    return [df[df['patientunitstayid'].isin(common_patients)] for df in dataframes]

def preprocess_data(patients, labels, labs, ncs, diags,test):
    patients, labels, labs, ncs, diags = filter_common_patients([patients, labels, labs, ncs, diags], test=test)
    patients, labs, ncs = utils.getLimitTimedata(patients, labs, ncs)
    patients, labels, labs, ncs, diags = filter_common_patients([patients, labels, labs, ncs, diags],test)
    return patients, labels, labs, ncs, diags

def save_dataframes(dataframes, output_dir, filenames):
    for df, filename in zip(dataframes, filenames):
        print(f'==> Saving the {filename} table...')
        df.to_csv(os.path.join(output_dir, filename), index=False)
def data_extraction_root(args,test = False):
    output_dir = os.path.join(args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    eicu_dir = os.path.join(args.eicu_dir)
    if not os.path.exists(eicu_dir):
        os.mkdir(eicu_dir)

    patients = utils.read_patients_table(args.eicu_dir)
    labels = utils.read_labels_table(args.eicu_dir)
    labs = utils.read_lab_table(args.eicu_dir)
    ncs = utils.read_nc_table(args.eicu_dir)
    diags = utils.read_diagnosis_table(args.eicu_dir)

    patients, labels, labs, ncs, diags = preprocess_data(patients, labels, labs, ncs, diags,test)

    output_dir = args.output_dir
    save_dataframes([patients, labels, labs, ncs, diags], output_dir,
                    ['all_flats.csv', 'all_labels.csv', 'all_labs.csv', 'all_ncs.csv', 'all_diags.csv'])


def main():
    parser = argparse.ArgumentParser(description="Create data for root")
    parser.add_argument('--eicu_dir',default='original_data/', type=str, help="Path to root folder containing all the patietns data")
    parser.add_argument('--output_dir', default='processed_data/',type=str, help="Directory where the created data should be stored.")
    parser.add_argument('--fl_dir', default='fl_data/', type=str,help="Directory where the created fl data should be stored.")
    parser.add_argument('--final_dir', default='../final_dataset/', type=str,help="Directory where the final fl data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.fl_dir):
        os.makedirs(args.fl_dir)
    if not os.path.exists(args.fl_dir):
        os.makedirs(args.final_dir)

    #data_process
    print('==> Removing the stays.txt file if it exists...')
    try:
        os.remove(args.output_dir + 'stays.txt')
    except FileNotFoundError:
        pass
    #数据提取
    data_extraction_root(args,test = False)
    # # 数据处理
    cut_off_prevalence = 0.01  # this would be 1%
    utils.timeseries_main(args.output_dir, test=False)
    utils.diagnoses_main(args.output_dir, cut_off_prevalence)
    utils.flat_and_labels_main(args.output_dir)
    # 数据分配
    utils.fl_hos_pro(args.output_dir,args.fl_dir)
    utils.move_choosed_fl_data(args.fl_dir,args.final_dir)

if __name__ == '__main__':
    main()