import argparse
import multiprocessing
import os
import subprocess
from tqdm import tqdm
import glob
import io
import csv
import json
from collections import defaultdict
import locale
import platform
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from terminaltables import AsciiTable
import random
from bisect import bisect_left

def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer)
                            for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)

def simple_worker(args):
    """Function to be executed in a separate process"""
    jsonl_file = args["jsonl_file"]
    big_domain = args["big_domain"]
    sub_domain = args["sub_domain"]
    # count = wc_count(jsonl_file)
    line_count = iter_count(jsonl_file)
    # print(f"jsonl_file:{jsonl_file}, count:{count}")
    return {'jsonl_file': jsonl_file, 'line_count': line_count, "big_domain": big_domain, "sub_domain": sub_domain}


def detail_worker(args):
    """Function to be executed in a separate process"""
    if 'result' in args:
        return args['result']
    else:
        jsonl_file = args["jsonl_file"]
        big_domain = args["big_domain"]
        sub_domain = args["sub_domain"]
        sample_length_calculator = args["sample_length_calculator"]
        length_sample_nums = args["length_sample_nums"]
        character_counts = 0
        byte_count = 0
        line_count = 0
        sample_lengths = {}
        
        sample_datas = [[] for i in range(len(sample_length_calculator.bin_names))]
        with open(jsonl_file, 'r') as file:
            for line in file:
                if line == "\n":
                    continue
                # Load each line as a JSON object and append to the data list
                try:
                    data = json.loads(line)
                except Exception as e:
                    e_type, e_value, e_traceback = sys.exc_info()
                    print("type ==> %s" % (e_type.__name__))
                    print("value ==> %s" % (e_value))
                    print(
                        f"error read jsonl_file:{jsonl_file}, line_count:{line_count}, line:{line[:100]}")
                    continue
                line_count += 1
                # print(f"line:{line}")
                # print(f"data['content']:{data['content']}")
                try:
                    character_counts += len(data["content"])
                    utf8_bytes = data['content'].encode('utf-8')
                    byte_count += len(utf8_bytes)
                except:
                    #print(f"xxxxxx data:{data}")
                    if "prompt" not in data:
                        assert 0, f"xxxxxx data:{data}, jsonl_file:{jsonl_file}"
                    if "output" not in data:
                        assert 0, f"xxxxxx data:{data}, jsonl_file:{jsonl_file}"
                    character_counts += len(data["prompt"])
                    utf8_bytes = data['prompt'].encode('utf-8')
                    byte_count += len(utf8_bytes)
                    character_counts += len(data["output"])
                    utf8_bytes = data['output'].encode('utf-8')
                    byte_count += len(utf8_bytes)
                if "content" in data:
                    sample_lengths[data["id"]] = len(data["content"])
                    _, bin_index = sample_length_calculator.get_sample_estimate_tokens(len(data["content"]), big_domain)
                    if length_sample_nums > 0 and random.random() > 0.7:
                        sample_data = dict(
                            content=data["content"],
                            id=data["id"],
                            big_domain=big_domain,
                            sub_domain=sub_domain,
                            file_name=os.path.basename(jsonl_file)
                        )
                        sample_datas[bin_index].append(sample_data)
        interval_stas = None
        if "content" in data:
            sample_lengths_list = list(sample_lengths.values())
            interval_stas = sample_length_calculator.calcalute(sample_lengths_list, big_domain)

        # print(f"jsonl_file:{jsonl_file}, count:{count}")
        return {'jsonl_file': os.path.basename(jsonl_file), 'line_count': line_count, "big_domain": big_domain, "sub_domain": sub_domain, "character_count": character_counts, "interval_stas": interval_stas, "sample_datas": sample_datas, "byte_count": byte_count}


def gen_query(jsonl_file_all_paths, existing_data, sample_length_calculator, length_sample_nums):
    for jsonl_file_all_path in jsonl_file_all_paths:
        jsonl_file = os.path.basename(jsonl_file_all_path)
        file_dir = os.path.dirname(jsonl_file_all_path)
        sub_domain = os.path.basename(file_dir)
        big_domain = os.path.basename(os.path.dirname(file_dir))
        sub_jsonl_path = f"{big_domain}/{sub_domain}/{jsonl_file}"

        if "jsonl" in os.path.splitext(jsonl_file_all_path)[-1]:
            if sub_jsonl_path in existing_data:
                yield {'result': existing_data[sub_jsonl_path]}
            else:
                yield {
                    "jsonl_file": jsonl_file_all_path,
                    "big_domain": big_domain,
                    "sub_domain": sub_domain,
                    "sample_length_calculator": sample_length_calculator,
                    "length_sample_nums": length_sample_nums
                }


def write_csv_buffer(buffer, fout, writer):
    if fout is not None:
        for d in buffer:
            writer.writerow(d)
            fout.flush()


def write_buffer(buffer, fout):
    if fout is not None:
        for d in buffer:
            fout.write(json.dumps(d) + "\n")
            fout.flush()

def write_to_jsonlines_file(file_path, data_list, mode="w"):
    with open(file_path, mode, encoding='utf-8') as file:
        for data in data_list:
            # 将每个字典转换为JSON字符串，并添加换行符
            json.dump(data, file, ensure_ascii=False)
            file.write('\n')


def read_jsonlines_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Load each line as a JSON object and append to the data list
            data.append(json.loads(line))
    return data


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        # 同时显示数值和占比的饼图
        return '{p:.1f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

def plot_sample_lengths(results, results_nums, output_dir, output_name):
    # bins = [0, 100, 500, 1000, 2048, 4096, 8192,
    #         16384, 20000, 32000, 10000000]  # 分组的依据
    # bin_names = ["0-100", "100-500", "500-1000", "1000-2048", "2048-4096",
    #              "4096-8192", "8192-16384", "16384-20000", "20000-32000", "32000-10000000"]
    # interval_stas = defaultdict(int)

    cn_total_interval_stas = defaultdict(int)
    en_total_interval_stas = defaultdict(int)
    for result in tqdm(results, total=results_nums):
        if result["big_domain"] == "cn":
            total_interval_stas = cn_total_interval_stas
        else:
            total_interval_stas = en_total_interval_stas
            
        for key, value in result["interval_stas"].items():
            total_interval_stas[key] += value
    stas_values = np.array(list(cn_total_interval_stas.values()))
    stats_ratio = stas_values / float(np.sum(stas_values)) * 100
    cur_result = []
    interval_stas_keys = list(cn_total_interval_stas.keys())
    for i in range(stas_values.shape[0]):
        cur_result.append([interval_stas_keys[i], stas_values[i], np.round(stats_ratio[i], 3)])
    table_data = [['range', 'estimate_token_nums', "ratio(%"]]
    table_data.extend(cur_result)
    table = AsciiTable(table_data)
    print(f"cn_total_interval_stas")    
    print(table.table)    
    plt.pie(stas_values, labels=cn_total_interval_stas.keys(), autopct=make_autopct(stas_values))
    pie_path = f"{output_dir}/{output_name}_cn_num_pie.png"
    plt.savefig(pie_path)

    stas_values = np.array(list(en_total_interval_stas.values()))
    stats_ratio = stas_values / float(np.sum(stas_values)) * 100
    cur_result = []
    for i in range(stas_values.shape[0]):
        cur_result.append([interval_stas_keys[i], stas_values[i], np.round(stats_ratio[i], 3)])
    table_data = [['range', 'estimate_token_nums', "ratio(%"]]
    table_data.extend(cur_result)
    table = AsciiTable(table_data)
    print(f"en_total_interval_stas")    
    print(table.table)    

    print(f"cn_total_interval_stas:{cn_total_interval_stas}")
    print(f"en_total_interval_stas:{en_total_interval_stas}")
    # print(f"en_total_interval_stas ratio(%) :{stas_values / float(np.sum(stas_values)) * 100}")
    plt.pie(stas_values, labels=en_total_interval_stas.keys(), autopct=make_autopct(stas_values))
    pie_path = f"{output_dir}/{output_name}_en_num_pie.png"
    plt.savefig(pie_path)
    # plt.cla()
    # plt.pie(stas_ratios, labels=interval_stas.keys())
    # pie_path = f"{output_dir}/{output_name}_ratio_pie.png"
    # plt.savefig(pie_path)

class SampleLengthCalculator(object):
    def __init__(self, bins, bin_names) -> None:
        '''
        description: 
        param {*} self
        param {*} bins
        param {*} bin_names
        return {*}
        '''
        self.bins = bins
        self.bin_names = bin_names
        self.charactor_to_token_ratios = {
            "cn": 1,
            "en": 0.25,
            "other": 0.25
        }
    
    def calcalute(self, sample_lengths, domain):
        interval_stas = defaultdict(int)
        sample_lengths_df = pd.Series(sample_lengths)
        if domain in self.charactor_to_token_ratios:
            charactor_to_token_ratio = self.charactor_to_token_ratios[domain]
        else:
            charactor_to_token_ratio = self.charactor_to_token_ratios["other"]
        # 将字符长度大致转成token长度
        sample_lengths_df = sample_lengths_df * float(charactor_to_token_ratio)  
        sample_df = pd.cut(sample_lengths_df, self.bins, labels=self.bin_names)
        cur_interval_stas_dcit = sample_df.value_counts().to_dict()
        for key, value in cur_interval_stas_dcit.items():
            interval_stas[key] += value
        return interval_stas
    
    def get_sample_estimate_tokens(self, sample_length, domain):
        if domain in self.charactor_to_token_ratios:
            charactor_to_token_ratio = self.charactor_to_token_ratios[domain]
        else:
            charactor_to_token_ratio = self.charactor_to_token_ratios["other"]
        sample_tokens = sample_length * float(charactor_to_token_ratio)
        # 计算当前采样数据在哪个bin
        bin_index = max(bisect_left(self.bins, sample_tokens) - 1, 0)
        return sample_tokens, bin_index


    # def plot_pie(self, interval_stas, pie_path):
    #     stas_values = np.array(list(interval_stas.values()))
    #     plt.pie(stas_values, labels=interval_stas.keys(), autopct=make_autopct(stas_values))
    #     #pie_path = f"{output_dir}/{output_name}_num_pie.png"
    #     plt.savefig(pie_path)


def main():
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--input_path', help='Help message for arg_name')
    parser.add_argument('--subdomain', default="", help='Help message for arg_name')
    parser.add_argument('--output_dir', help='Help message for arg_name')
    parser.add_argument('--output_name', help='Help message for arg_name')
    parser.add_argument('--num_processes', type=int,
                        default=10, help='Help message for arg_name')
    parser.add_argument('--length_sample_nums', type=int, default=0, help='Help message for arg_name')
    parser.add_argument('--detail_report', action='store_true')
    args = parser.parse_args()

    num_processes = args.num_processes
    input_path = args.input_path
    subdomain = args.subdomain
    output_dir = args.output_dir
    output_name = args.output_name
    detail_report = args.detail_report
    length_sample_nums = args.length_sample_nums

    output_csv = f"{output_name}.csv"
    output_jsonl = f"{output_name}.jsonl"
    output_jsonl_path = f"{output_dir}/{output_jsonl}"
    output_csv_path = f"{output_dir}/{output_csv}"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if subdomain:
        file_paths = glob.glob(f"{input_path}/**/{subdomain}/*.jsonl", recursive=True)
    else:
        file_paths = glob.glob(f"{input_path}/**/*.jsonl", recursive=True)
    print(f"find {len(file_paths)} jsonls")

    # load from existing
    existing_data = {}
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                total_path = f"{instance['big_domain']}/{instance['sub_domain']}/{instance['jsonl_file']}"
                existing_data[total_path] = instance
        print(
            f'load {len(existing_data)} examples from existing processed data........................')
    
    # 统计token长度分级
    bins = [0, 100, 500, 1000, 2048, 4096, 8192, 16384, 20000, 32000, 10000000]  # 分组的依据
    bin_names = ["0-100", "100-500", "500-1000", "1000-2048", "2048-4096",
                 "4096-8192", "8192-16384", "16384-20000", "20000-32000", "32000-10000000"]
    sample_length_calculator = SampleLengthCalculator(bins, bin_names)

    pool = multiprocessing.Pool(processes=num_processes)
    if not detail_report:
        results = pool.imap_unordered(
            simple_worker, gen_query(file_paths, existing_data, sample_length_calculator, length_sample_nums))
    else:
        results = pool.imap_unordered(
            detail_worker, gen_query(file_paths, existing_data, sample_length_calculator, length_sample_nums))

    json_buffer = []
    buffer_size = 100

    if output_jsonl_path is not None:
        fout_jsonl = open(output_jsonl_path, 'w')
    else:
        fout_jsonl = None

    if length_sample_nums == 0:
        for result in tqdm(results, total=len(file_paths)):
            json_buffer.append(result)
            if len(json_buffer) > buffer_size:
                write_buffer(json_buffer, fout_jsonl)
                json_buffer = []
        if len(json_buffer) > 0:
            write_buffer(json_buffer, fout_jsonl)
        if fout_jsonl is not None:
            fout_jsonl.close()
    else:
        total_sample_datas = [0 for i in range(len(bin_names))]

        for bin_index, bin_name in enumerate(bin_names):
            jsonl_path = f"{output_dir}/{output_name}_sample_{bin_name}.jsonl"
            fout_jsonl = open(jsonl_path, 'w')
            fout_jsonl.close()
        # 仅采样数据
        if length_sample_nums > 0:
            ret_index = 0
            for result in tqdm(results, total=len(file_paths)):
                sample_datas = result["sample_datas"]
                all_length_sample_finish = True
                for bin_index, sapmle_data in enumerate(sample_datas):
                    if total_sample_datas[bin_index] < length_sample_nums:
                        all_length_sample_finish = False
                        total_sample_datas[bin_index] += len(sapmle_data)
                        bin_name = bin_names[bin_index]
                        jsonl_path = f"{output_dir}/{output_name}_sample_{bin_name}.jsonl"
                        write_to_jsonlines_file(jsonl_path, sample_datas[bin_index], mode="a")
                        print(f"sample_datas ret_index: {ret_index} index:{bin_index}, bin_name:{bin_names[bin_index]}, nums: {total_sample_datas[bin_index]}")
                # for bin_index, bin_name in enumerate(bin_names):
                #     jsonl_path = f"{output_dir}/{output_name}_sample_{bin_name}.jsonl"
                #     if ret_index == 0:
                #         mode = "w"
                #     else:
                #         mode = "a"
                #     write_to_jsonlines_file(jsonl_path, sample_datas[bin_index], mode=mode)
                if all_length_sample_finish:
                    break
                ret_index += 1
            print(f"successfuly get samples!!!!!!!!!!!!!!")
            sys.exit(0)
            
    if output_csv_path is not None:
        fout_csv = io.open(output_csv_path, 'w', encoding='utf-8')
    else:
        fout_csv = None
    writer = csv.writer(fout_csv)

    if not detail_report:
        write_csv_buffer(
            [["big_domain", "sub_domain", "samples_nums", "file_nums"]], fout_csv, writer)
    else:
        write_csv_buffer([["big_domain", "sub_domain", "samples_nums",
                         "file_nums", "character_nums", "byte_nums"]], fout_csv, writer)

    results = read_jsonlines_file(output_jsonl_path)

    csv_buffer = []
    domain_statistics = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))
    for result in tqdm(results, total=len(file_paths)):
        # print(f"result:{result}")
        domain_statistics[result["big_domain"]
                          ][result["sub_domain"]]["file_nums"] += 1
        domain_statistics[result["big_domain"]][result["sub_domain"]
                                                ]["samples_nums"] += result["line_count"]
        if "character_count" in result:
            domain_statistics[result["big_domain"]][result["sub_domain"]
                                                    ]["character_nums"] += result["character_count"]
        if "byte_count" in result:
            domain_statistics[result["big_domain"]][result["sub_domain"]
                                                    ]["byte_nums"] += result["byte_count"]
            

    for big_domain, big_domain_dict in domain_statistics.items():
        for sub_domain, sub_domain_dict in big_domain_dict.items():
            if not detail_report:
                csv_buffer.append(
                    [big_domain, sub_domain, sub_domain_dict["samples_nums"], sub_domain_dict["file_nums"]])
            else:
                csv_buffer.append([big_domain, str(sub_domain), sub_domain_dict["samples_nums"],
                                  sub_domain_dict["file_nums"], sub_domain_dict["character_nums"], sub_domain_dict["byte_nums"]])

    write_csv_buffer(csv_buffer, fout_csv, writer)

    if fout_csv is not None:
        fout_csv.close()

    if detail_report:
        plot_sample_lengths(results, len(file_paths), output_dir, output_name)


if __name__ == '__main__':
    main()
