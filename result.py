import os
import shutil

import pandas as pd

def calculate_weighted_avg(result_dict, num_dict):
    """计算加权平均值的辅助函数。"""
    weights = [num_dict.get(str(key), 0) for key in result_dict.keys()]
    weighted_values = [v * num_dict.get(str(k), 0) for k, v in result_dict.items()]
    if sum(weights) > 0:
        return sum(weighted_values) / sum(weights)
    else:
        print("警告: 权重总和为0，返回0作为加权平均值")
        return 0

    # 获取均值和标准差


def getMeanAndStd(result_list):
    result_df = pd.DataFrame(result_list)
    means = result_df.mean()
    stds = result_df.std()
    result = {}
    for col in means.index:
        result[str(col) + '_mean'] = means[col]  # 强制转换为字符串
        result[str(col) + '_std'] = stds[col]  # 强制转换为字符串
    return result
def mort_table(path, num_dict):
    result_df = pd.DataFrame()  # 初始化结果 DataFrame

    # 处理 AUROC 指标
    def process_indicator(indicator):
        indicator_result_df = pd.DataFrame()  # 初始化每个指标的结果 DataFrame
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                result_list = []
                seed_flolder = os.listdir(folder_path)
                # 当子文件夹数量小于 5 时，跳过当前文件夹
                if len(seed_flolder) < 5:
                    continue

                for sub_folder in seed_flolder:
                    result_path = os.path.join(folder_path, sub_folder, 'Final_Mort_results.csv')
                    if os.path.exists(result_path):  # 确保文件存在
                        try:
                            df = pd.read_csv(result_path)
                            df = df.tail(len(num_dict))  # 选择最后 len(num_dict) 行

                            # 提取当前指标列并计算加权平均
                            if indicator in df.columns and 'center' in df.columns:
                                result_dict = df.set_index('center')[indicator].to_dict()
                                avg_value = calculate_weighted_avg(result_dict, num_dict)
                                result_dict['Avg'] = avg_value
                                result_list.append(result_dict)
                            else:
                                print(f"文件 {result_path} 缺少必要的列：'center' 或 '{indicator}'")
                        except Exception as e:
                            print(f"读取文件 {result_path} 时出错: {e}")

                if result_list:  # 确保 result_list 非空
                    result = getMeanAndStd(result_list)
                    result['alg'] = folder
                    df_temp = pd.DataFrame([result])  # 使用列表创建 DataFrame
                    indicator_result_df = pd.concat([indicator_result_df, df_temp],
                                                    ignore_index=True)  # 将结果追加到指标的 DataFrame

        # 将 'alg' 列移到 DataFrame 的开头
        if not indicator_result_df.empty:
            alg_column = indicator_result_df.pop('alg')
            indicator_result_df.insert(0, 'alg', alg_column)

            # 创建新的 DataFrame 格式（包括 mean 和 std）
            columns = indicator_result_df.columns.tolist()
            columns_mean = [col for col in columns if col.endswith('_mean')]
            columns_std = [col for col in columns if col.endswith('_std')]

            new_columns = ['alg'] + [col.replace('_mean', '') for col in columns_mean]
            final_df = pd.DataFrame(columns=new_columns)

            # 填充 final_df，其中每个算法的均值和标准差分成两行
            for i, row in indicator_result_df.iterrows():
                alg_mean_row = [row['alg'] + '_mean']
                alg_std_row = [row['alg'] + '_std']
                for col in columns_mean:
                    alg_mean_row.append(row[col])
                for col in columns_std:
                    alg_std_row.append(row[col])
                final_df.loc[len(final_df)] = alg_mean_row
                final_df.loc[len(final_df)] = alg_std_row

            return final_df
        else:
            return pd.DataFrame()  # 返回空的 DataFrame 如果没有有效的数据

    # 只处理 AUROC 指标
    indicator = 'auroc'
    result_dict = {indicator: process_indicator(indicator)}

    # 使用 ExcelWriter 将结果保存到不同的工作表
    output_file = os.path.join(path, 'mort_result.xlsx')
    with pd.ExcelWriter(output_file) as writer:
        for indicator, final_df in result_dict.items():
            if not final_df.empty:
                final_df.to_excel(writer, sheet_name=indicator, index=False)
    print(f"结果已保存至 {output_file}")

def LoS_table(path, num_dict):
    def process_indicator(indicator):
        indicator_result_df = pd.DataFrame()  # 初始化每个指标的结果 DataFrame
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                result_list = []
                # seed_folder = [str(item) for item in os.listdir(folder_path) if   item != '15']
                seed_folder =  os.listdir(folder_path)
                print(seed_folder)

                for sub_folder in seed_folder :
                    # print(sub_folder)
                    result_path = os.path.join(folder_path, sub_folder, 'Final_Los_results.csv')
                    if os.path.exists(result_path):  # 确保文件存在
                        try:
                            df = pd.read_csv(result_path)

                            df = df.tail(len(num_dict))  # 选择最后 len(num_dict) 行

                            # 提取当前指标列并计算加权平均
                            if indicator in df.columns and 'center' in df.columns:
                                result_dict = df.set_index('center')[indicator].to_dict()
                                avg_value = calculate_weighted_avg(result_dict, num_dict)
                                result_dict['Avg'] = avg_value
                                result_list.append(result_dict)
                            else:
                                print(f"文件 {result_path} 缺少必要的列：'center' 或 '{indicator}'")
                        except Exception as e:
                            print(f"读取文件 {result_path} 时出错: {e}")

                if result_list:  # 确保 result_list 非空
                    result = getMeanAndStd(result_list)
                    result['alg'] = folder
                    df_temp = pd.DataFrame([result])  # 使用列表创建 DataFrame
                    indicator_result_df = pd.concat([indicator_result_df, df_temp],
                                                    ignore_index=True)  # 将结果追加到指标的 DataFrame

        # 将 'alg' 列移到 DataFrame 的开头
        if not indicator_result_df.empty:
            alg_column = indicator_result_df.pop('alg')
            indicator_result_df.insert(0, 'alg', alg_column)

            # 创建新的 DataFrame 格式（包括 mean 和 std）
            columns = indicator_result_df.columns.tolist()
            columns_mean = [col for col in columns if col.endswith('_mean')]
            columns_std = [col for col in columns if col.endswith('_std')]

            new_columns = ['alg'] + [col.replace('_mean', '') for col in columns_mean]
            final_df = pd.DataFrame(columns=new_columns)

            # 填充 final_df，其中每个算法的均值和标准差分成两行
            for i, row in indicator_result_df.iterrows():
                alg_mean_row = [row['alg'] + '_mean']
                alg_std_row = [row['alg'] + '_std']
                for col in columns_mean:
                    alg_mean_row.append(row[col])
                for col in columns_std:
                    alg_std_row.append(row[col])
                final_df.loc[len(final_df)] = alg_mean_row
                final_df.loc[len(final_df)] = alg_std_row

            return final_df
        else:
            return pd.DataFrame()  # 返回空的 DataFrame 如果没有有效的数据

    # 处理每个指标（MSE, MAPE, MAD, R2）
    result_dict = {}
    for indicator in [ 'rmse']:
        result_dict[indicator] = process_indicator(indicator)

    # 使用 ExcelWriter 将结果保存到不同的工作表
    output_file = os.path.join(path, 'LoS_result.xlsx')
    with pd.ExcelWriter(output_file) as writer:
        for indicator, final_df in result_dict.items():
            if not final_df.empty:
                final_df.to_excel(writer, sheet_name=indicator, index=False)

    print(f"结果已保存至 {output_file}")


def read_and_output_avg_mort(excel_file):
    data = pd.read_excel(excel_file)
    algorithm_data = {}
    avg_values = data['Avg']
    algorithm_names = data['alg']
    for idx, alg_name in enumerate(algorithm_names):
        if 'mean' in alg_name:
            alg = alg_name.replace('_mean', '')
            mean_value = avg_values[idx]
            std_value = avg_values[idx + 1]
            value_str = f"{mean_value:.4f} $\pm$ {std_value:.4f}"
            if alg not in algorithm_data:
                algorithm_data[alg] = []
            algorithm_data[alg].append(value_str)
            # 输出合并后的数据
    for alg, values in algorithm_data.items():
        output_str = f"{alg}"
        for value in values:
            output_str += f" & {value}"
        print(output_str)


def read_and_output_avg_LoS(excel_file):
    # 读取 Excel 文件
    with pd.ExcelFile(excel_file) as xls:
        # 读取每个工作表
        # mse_df = pd.read_excel(xls, sheet_name='mse')
        rmse_df  = pd.read_excel(xls, sheet_name='rmse')


    # 存储每个算法的不同指标数据
    algorithm_data = {}
    sheets = [ rmse_df]

    for data in sheets:
        avg_values = data['Avg']
        algorithm_names = data['alg']

        for idx, alg_name in enumerate(algorithm_names):
            if 'mean' in alg_name:
                alg = alg_name.replace('_mean', '')
                mean_value = avg_values[idx]
                std_value = avg_values[idx + 1]
                value_str = f"{mean_value:.4f} $\pm$ {std_value:.4f}"

                if alg not in algorithm_data:
                    algorithm_data[alg] = []
                algorithm_data[alg].append(value_str)

    # 输出合并后的数据
    for alg, values in algorithm_data.items():
        output_str = f"{alg}"
        for value in values:
            output_str += f" & {value}"
        print(output_str)

if __name__ == '__main__':


    num_dict = {'264': 1346, '420': 981, '443': 911, '458': 858, '338': 836, '252': 718, '122': 706, '300': 691,'188': 685, '73': 683}



    source_path = 'Results'
    mort_path = os.path.join(source_path, 'mortality')
    mort_excel_file =  os.path.join(mort_path,'mort_result.xlsx')
    mort_table(mort_path, num_dict)




