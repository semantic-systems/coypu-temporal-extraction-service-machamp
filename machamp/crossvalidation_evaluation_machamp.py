#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import re
from typing import Dict, List, Tuple, Any
import json
from machamp.crossval.scorer import *
import difflib
import pprint
import math
import sys
import shutil
import pandas as pd
import argparse


MODEL_FILE_REGEX = r"model(_[0-9]+)*.pt" #e.g. model.pt, model_1.pt, model_2.pt
MODEL_CROSSVALIDATION_FOLDER_SUFFIX_REGEX = r"cv_fold_[0-9]{1}" #e.g. cv_fold_1, cv_fold_2, cv_fold_3
VAL_RESULT_FILE_SUFFIX = r".out"
TEST_RESULT_FILE_SUFFIX = r"model_[0-9]+\.out\.test$" #e.g. model_1.out.test, model_2.out.test, model_3.out.test

RELEVANT_DATA_KEYS_EVALTYPE: list = ["strict", "relaxed"]
RELEVANT_DATA_KEYS_TARGETTYPE: list = ["span", "typespan"]
RELEVANT_DATA_KEYS_TEMPTYPE: list = ["total", "tempexp", "date", "time", "duration", "set"]
RELEVANT_DATA_KEYS_DATATYPE: list = ["string", "offset"]
RELEVANT_DATA_KEYS_METRICTYPE: list = ["F1", "P", "R"]

RELEVANT_COMBINATIONS = [
    f"{evaltype}_{targettype}_{temptype}_{datatype}_{metrictype}" 
    for evaltype in RELEVANT_DATA_KEYS_EVALTYPE 
    for targettype in RELEVANT_DATA_KEYS_TARGETTYPE 
    for temptype in RELEVANT_DATA_KEYS_TEMPTYPE 
    for datatype in RELEVANT_DATA_KEYS_DATATYPE 
    for metrictype in RELEVANT_DATA_KEYS_METRICTYPE
]



def _write_best_models_to_file(
        directory_path: str,
        model_full_name: str, 
        best_models_val: List[Tuple[str, str, str, dict, list, list]],
        best_models_f1_val: List[float],
        best_models_test: List[Tuple[str, str, str, dict, list, list]],
        best_models_f1_test: List[float]
    ) -> None:
    """
    Writes the overall best model results (test & val) to file.

    Args:
        directory_path (str): Directory to save the file to.
        model_full_name (str): Full name of the model.
        best_models_val (List[Tuple[str, str, str, dict, list, list]]):
            List of tuples containing the results for each model.
        best_models_f1_val (List[float]): List of f1 scores for each model.
        best_models_test (List[Tuple[str, str, str, dict, list, list]]):
            List of tuples containing the results for each model.
        best_models_f1_test (List[float]): List of f1 scores for each model.

    """
    assert(len(best_models_val) == len(best_models_f1_val))
    assert(len(best_models_test) == len(best_models_f1_test))

    filepath = os.path.join(directory_path, f"{model_full_name}_crossvalidation_best_models.txt")
    with open(filepath, "w") as output:
        print(f"Writing to {output.name}")
        checkpoint_regex = re.compile(f".*checkpoint-(\d+)")
        output.write(f"Best models for {model_full_name} based on 'val' gold dataset:\n")
        for i, model in enumerate(best_models_val):
            #If model_path ends on regex "checkpoint_\d+" then extract the checkpoint together with the number
            model_path = model[2]
            match = re.search(checkpoint_regex, model_path)
            checkpoint_name = "regular"
            if match:
                checkpoint_name = "checkpoint-" + match.group(1)
            output.write(f"({i+1})   {model[0]}_{model[1]}, {checkpoint_name} Total String F1: {best_models_f1_val[i]}\n{model[2]}\n\n")

        output.write("\n\n\n")

        output.write(f"\nBest models for {model_full_name} based on 'test' gold dataset:\n")
        for i, model in enumerate(best_models_test):
            #If model_path ends on regex "checkpoint_\d+" then extract the checkpoint together with the number
            model_path = model[2]
            match = re.search(checkpoint_regex, model_path)
            checkpoint_name = "regular"
            if match:
                checkpoint_name = "checkpoint-" + match.group(1)
            output.write(f"({i+1})   {model[0]}_{model[1]}, {checkpoint_name} Total String F1: {best_models_f1_test[i]}\n{model[2]}\n\n")



def _copy_gold_files(gold_val_files: List[str], gold_test_files: List[str], output_directory: str, model_full_name: str) -> None:
    """
    Copies the gold val and test files to the logfile directory.
    Files are copied for convenience of proximity to the logfiles.
    They are fully redundant and serve no other purpose as described above.

    Args:
        gold_val_files (List[str]): List of full filepaths to the gold val files.
        gold_test_files (List[str]): List of full filepaths to the gold test files.
        output_directory (str): Directory to copy the files to.
        model_full_name (str): Full name of the model.
    """
    for gold_val_file, gold_test_file in zip(gold_val_files, gold_test_files):
        val_filename = gold_val_file.split("/")[-2]
        test_filename = gold_test_file.split("/")[-2]
        val_filepath = os.path.join(output_directory, f"_{val_filename}_gold_val.json")
        test_filepath = os.path.join(output_directory, f"_{test_filename}_gold_test.json")
        shutil.copy(gold_val_file, val_filepath)
        shutil.copy(gold_test_file, test_filepath)



def _write_best_fold_models_to_file(
        directory_path: str, 
        model_full_name: str, 
        best_models_val: dict, 
        best_models_f1_val: dict, 
        best_models_test: dict, 
        best_models_f1_test: dict
    ) -> None:
    """
    For each fold it writes the best model to file (for both val and test). 
    That means that there could only exist one model per fold.

    Args:
        directory_path (str): Directory to save the file to.
        model_full_name (str): Full name of the model.
        best_models_val (dict): Dictionary containing the best 
            model for each fold for the val predictions.
        best_models_f1_val (dict): Dictionary containing the f1 score
            for each fold for the val predictions.
        best_models_test (dict): Dictionary containing the best
            model for each fold for the test predictions.
        best_models_f1_test (dict): Dictionary containing the f1 score
            for each fold for the test predictions.
    """
    output_filepath = os.path.join(directory_path, f"{model_full_name}_crossvalidation_best_fold_models.txt")
    best_val = list()
    for i, (fold, results) in enumerate(best_models_val.items()):
        best_val += [(best_models_f1_val[fold], results)]
    best_val_sorted = sorted(best_val, key=lambda x: x[0], reverse=True)

    best_test = list()
    for i, (fold, results) in enumerate(best_models_test.items()):
        best_test += [(best_models_f1_test[fold], results)]
    best_test_sorted = sorted(best_test, key=lambda x: x[0], reverse=True)

    #Sort the models based on the f1 score

    with open(output_filepath, "w") as output:
        print(f"Writing to {output.name}")
        checkpoint_regex = re.compile(f".*checkpoint-(\d+)")

        output.write(f"Best fold-model for {model_full_name} based on 'val' gold-dataset:\n")
        output.write("-" * 60 + "\n\n")

        for i, item in enumerate(best_val_sorted):
            f1, (model_full_name, fold, model_path, results, _, _) = item
            #If model_path ends on regex "checkpoint_\d+" then extract the checkpoint together with the number
            match = re.search(checkpoint_regex, model_path)
            checkpoint_name = "regular"
            if match:
                checkpoint_name = "checkpoint-" + match.group(1)
                
            output.write(f"({i+1})   {model_full_name}_{fold}, {checkpoint_name} Total String F1: {f1}\n{model_path}\n\n")

        output.write("\n\n\n")

        output.write(f"Best fold-model for {model_full_name} based on 'test' gold-dataset:\n")
        output.write("-" * 75 + "\n\n")

        for i, item in enumerate(best_test_sorted):
            f1, (model_full_name, fold, model_path, results, _, _) = item
            #If model_path ends on regex "checkpoint_\d+" then extract the checkpoint together with the number
            match = re.search(checkpoint_regex, model_path)
            checkpoint_name = "regular"
            if match:
                checkpoint_name = "checkpoint-" + match.group(1)
            output.write(f"({i+1})   {model_full_name}_{fold}, {checkpoint_name} Total String F1: {f1}\n{model_path}\n\n")



def _create_models_structured_dataframe(model_directory_results, average_dict, std_dict, output_directory, eval_type) -> None:
    model_dataset_name = model_directory_results[0][0]

    header_row = [
        model_dataset_name, # e.g. TempEval

        "Strict-F1-Avg", "Strict-F1-Std",
        "Strict-P-Avg", "Strict-P-Std",
        "Strict-R-Avg", "Strict-R-Std",

        "Strict-Type-F1-Avg", "Strict-Type-F1-Std", 

        "Relaxed-F1-Avg", "Relaxed-F1-Std", 
        "Relaxed-P-Avg", "Relaxed-P-Std", 
        "Relaxed-R-Avg", "Relaxed-R-Std",
        
        "Relaxed-Type-F1-Avg", "Relaxed-Type-F1-Std"
    ]

    def _get_data_row(average_dict, std_dict, value):
        data_row = []
        data_row += [average_dict[f"strict_span_{value}_string_F1"], std_dict[f"strict_span_{value}_string_F1"]]
        data_row += [average_dict[f"strict_span_{value}_string_P"], std_dict[f"strict_span_{value}_string_P"]]
        data_row += [average_dict[f"strict_span_{value}_string_R"], std_dict[f"strict_span_{value}_string_R"]]
        data_row += [average_dict[f"strict_typespan_{value}_string_F1"], std_dict[f"strict_typespan_{value}_string_F1"]]
        data_row += [average_dict[f"relaxed_span_{value}_string_F1"], std_dict[f"relaxed_span_{value}_string_F1"]]
        data_row += [average_dict[f"relaxed_span_{value}_string_P"], std_dict[f"relaxed_span_{value}_string_P"]]
        data_row += [average_dict[f"relaxed_span_{value}_string_R"], std_dict[f"relaxed_span_{value}_string_R"]]
        data_row += [average_dict[f"relaxed_typespan_{value}_string_F1"], std_dict[f"relaxed_typespan_{value}_string_F1"]]
        return data_row
    
    def _get_temporal_class_data_row(average_dict, std_dict, value):
        data_row = []
        data_row += [average_dict[f"strict_typespan_{value}_string_F1"], std_dict[f"strict_typespan_{value}_string_F1"]]
        data_row += [average_dict[f"strict_typespan_{value}_string_P"], std_dict[f"strict_typespan_{value}_string_P"]]
        data_row += [average_dict[f"strict_typespan_{value}_string_R"], std_dict[f"strict_typespan_{value}_string_R"]]
        data_row += [average_dict[f"strict_typespan_{value}_string_F1"], std_dict[f"strict_typespan_{value}_string_F1"]]
        data_row += [average_dict[f"relaxed_typespan_{value}_string_F1"], std_dict[f"relaxed_typespan_{value}_string_F1"]]
        data_row += [average_dict[f"relaxed_typespan_{value}_string_P"], std_dict[f"relaxed_typespan_{value}_string_P"]]
        data_row += [average_dict[f"relaxed_typespan_{value}_string_R"], std_dict[f"relaxed_typespan_{value}_string_R"]]
        data_row += [average_dict[f"relaxed_typespan_{value}_string_F1"], std_dict[f"relaxed_typespan_{value}_string_F1"]]
        return data_row
    
    total_row = ["Total"]
    total_row += _get_data_row(average_dict, std_dict, "total")

    tempexp_row = ["TempExp"]
    tempexp_row += _get_temporal_class_data_row(average_dict, std_dict, "tempexp")

    date_row = ["Date"]
    date_row += _get_temporal_class_data_row(average_dict, std_dict, "date")

    time_row = ["Time"]
    time_row += _get_temporal_class_data_row(average_dict, std_dict, "time")
    
    duration_row = ["Duration"]
    duration_row += _get_temporal_class_data_row(average_dict, std_dict, "duration")

    set_row = ["Set"]
    set_row += _get_temporal_class_data_row(average_dict, std_dict, "set")

    df_data = [header_row, total_row, tempexp_row, date_row, time_row, duration_row, set_row]
    df = pd.DataFrame(df_data)

    output_filepath = os.path.join(output_directory, f"{model_dataset_name}_crossvalidation_full_evaluation_dataframe_{eval_type}.csv")
    df = df.round(2)
    pd.set_option('display.float_format', '{:.2f}'.format)

    print(f"Writing to Pandas Dataframe to {output_filepath}")
    df.to_csv(output_filepath)



def _write_model_results_to_files(model_directory_results, directory_path, evaluation_type, short_names=True):
    for fold, model_directory in enumerate(model_directory_results):
        model_full_name, fold_name, model_path, results, predict_records, predict_seq2seq = model_directory
        last_foldername = directory_path.split("/")[-1]
        original_model_path_as_name = model_path.split(f"{last_foldername}/")[-1]
        original_model_path_as_name = original_model_path_as_name.replace("/", "_")

        if short_names:
            checkpoint_regex = re.compile(f".*checkpoint-(\d+)")
            match = re.search(checkpoint_regex, original_model_path_as_name)
            checkpoint_name = ""
            if match:
                checkpoint_name = "checkpoint-" + match.group(1) 
                filepath = os.path.join(directory_path, f"{model_full_name}_{fold_name}_{checkpoint_name}")
            else:
                filepath = os.path.join(directory_path, f"{model_full_name}_{fold_name}")
        else:
            filepath = os.path.join(directory_path, f"{fold_name}_eval-results_{original_model_path_as_name}")


        with open(filepath + f"_{evaluation_type}_results.txt", "w") as output:
            print(f"Writing to {output.name}")
            for key, value in results.items():
                output.write(f'{key}={value}\n')

        with open(filepath + f"_{evaluation_type}_preds_record.txt", "w") as output:
            print(f"Writing to {output.name}")
            for record in predict_records:
                output.write(f'{json.dumps(record)}\n')



def _write_error_analysis_to_files(
    directory_path: str,
    evaluation_type: str,
    negative_cases: List[Dict[str, Any]],
    results: List[Tuple[str, str, str, dict, list, list]]
) -> None:
    """
    Writes the negative cases for each model to file.
    """
    assert len(negative_cases) == len(results)
    for i, model_directory in enumerate(results):
        model_full_name, fold_name, model_path, results, _, _ = model_directory
        negative_cases_for_model = negative_cases[i]
        
        checkpoint = ""
        last_dir = model_path.split("/")[-1]
        if "checkpoint" in last_dir:
            checkpoint = f"_{last_dir}".strip()

        filename = f"{model_full_name}_{fold_name}{checkpoint}_error_analysis_{evaluation_type}.txt"
        filepath = os.path.join(directory_path, filename)

        with open(filepath, "w") as output:
            print(f"Writing to {output.name}")
            output.write(f"ERROR ANALYSIS FOR {model_full_name}_{fold_name}{checkpoint} ON {evaluation_type}-dataset\n")
            output.write(len(f"ERROR ANALYSIS FOR {model_full_name}_{fold_name}{checkpoint} ON {evaluation_type}-dataset") * "-" + "\n\n")
            output.write(f"Error Report:\n")
            for case_type, negative_case in negative_cases_for_model.items():
                output.write(f"{case_type.upper()} errors: {len(negative_case)}\n")
            output.write(f"\n\n\n")


            for case_type, negative_case in negative_cases_for_model.items():
                output.write(f"{case_type.upper()} ANALYSIS ({len(negative_case)} errors):\n")
                output.write(len(f"{case_type.upper()} ANALYSIS ({len(negative_case)} errors):\n") * "-" + "\n")
                output.write(f"\n")

                for error_case in negative_case:
                    nc_index = error_case["index"]
                    nc_gold_text = error_case["gold_text"]
                    nc_gold_tokens = error_case["gold_tokens"]
                    nc_intersection = error_case["intersection"]
                    nc_gold_labeled_text = error_case["gold_labeled_text"]
                    nc_pred_labeled_text = error_case["pred_labeled_text"]
                    nc_gold_entities = error_case["gold_entities"]
                    nc_pred_entities = error_case["pred_entities"]

                    output.write(f"Error index in gold dataset: {nc_index}\n")
                    output.write(f"Gold text: {nc_gold_text}\n")
                    output.write(f"Gold tokens: {nc_gold_tokens}\n")
                    nc_gold_tokens_with_index = [f"{i}:{token}" for i, token in enumerate(nc_gold_tokens)]
                    output.write(f"Gold tokens with index: {nc_gold_tokens_with_index}\n")
                    output.write(f"Intersection (of entity strings): {nc_intersection}\n")
                    output.write(f"\n")
                    output.write(f"Gold labeled text: {nc_gold_labeled_text}\n")

                    output.write(f"Gold entities:\n")
                    for nc_gold_entity in nc_gold_entities:
                        output.write(f"\t{nc_gold_entity}\n")
                    output.write(f"\n")
                    output.write(f"Pred labeled text: {nc_pred_labeled_text}\n")
                    output.write(f"Pred entities:\n")
                    for nc_pred_entity in nc_pred_entities:
                        output.write(f"\t{nc_pred_entity}\n")
                    output.write(f"\n\n\n")
                output.write("-" * 150 + "\n")
                output.write("\n" * 10)



def _create_models_best_dataframe(model_directory_results, sum_dict, average_dict, variance_dict, std_dict, output_directory, data_type, filtering=False) -> None:
    """
    Creates a pandas dataframe for the best results i.e. top1 for each of the folds. The presented averages and std are calculated on these results.
    Saves the dataframe to csv.

    Args:
        model_directory_results (list): List of tuples containing the results for each model.
        sum_dict (dict): Dictionary containing the sum of all results for each metric.
        average_dict (dict): Dictionary containing the average of all results for each metric.
        variance_dict (dict): Dictionary containing the variance of all results for each metric.
        std_dict (dict): Dictionary containing the standard deviation of all results for each metric.
        output_directory (str): Directory to save the dataframe to.
        data_type (str): Either "val" or "test" depending on the data used to calculate the results.
    """
    data_headers = list()
    data_results = list()
    for model_directory_result in list(model_directory_results.values()):
        model_dataset_name, fold, model_fullname, results, _, _ = model_directory_result

        #If ends on regex "checkpoint_\d+" then extract the checkpoint together with the number
        checkpoint_regex = re.compile(f".*checkpoint-(\d+)")
        match = re.search(checkpoint_regex, model_fullname)
        checkpoint_name = ""
        if match:
            checkpoint_name = "_checkpoint-" + match.group(1)
        column_header = f"{fold}{checkpoint_name}"

        data_headers += [column_header]
        data_results += [results]

    average_header = f"AVERAGE_{model_dataset_name}"
    std_header = f"STD_{model_dataset_name}"
    variance_header = f"VARIANCE_{model_dataset_name}"
    sum_header = f"SUM_{model_dataset_name}"

    average_dict = dict(sorted(average_dict.items(), reverse=True))
    std_dict = dict(sorted(std_dict.items(), reverse=True))
    variance_dict = dict(sorted(variance_dict.items(), reverse=True))
    sum_dict = dict(sorted(sum_dict.items(), reverse=True))

    dict_list = [average_dict, std_dict, variance_dict, sum_dict] + data_results
    data_headers = [average_header, std_header, variance_header, sum_header] + data_headers

    if filtering:
        filtered_list = list()
        for dict_item in dict_list:
            filtered_dict = {}
            for key, value in dict_item.items():
                if key in RELEVANT_COMBINATIONS:
                    filtered_dict[key] = value
            filtered_list.append(filtered_dict)
        dict_list = filtered_list

    df = pd.DataFrame(dict_list).T
    df.columns = data_headers
    if filtering:
        output_filepath = os.path.join(output_directory, f"{model_dataset_name}_crossvalidation_dataframe_{data_type}_filtered.csv")
    else:
        output_filepath = os.path.join(output_directory, f"{model_dataset_name}_crossvalidation_dataframe_{data_type}.csv")
    print(f"Writing to Pandas Dataframe to {output_filepath}")
    df_rounded = df.round(4)
    df_rounded.to_csv(output_filepath)



def _write_f1_summary_to_file(
        model_directory_val_results: List[Tuple[str, str, str, dict, list, list]],
        model_directory_test_results: List[Tuple[str, str, str, dict, list, list]],
        directory_path: str
    ) -> None:
    """
    Writes the f1 summary to file. The summary contains all models and their f1 scores for both the val and test predictions.

    Args:
        model_directory_val_results (List[Tuple[str, str, str, dict, list, list]]):
            List of tuples containing the results for each val model.
        model_directory_test_results (List[Tuple[str, str, str, dict, list, list]]):
            List of tuples containing the results for each test model.
        directory_path (str): Directory to save the file to.

    """
    sort_metric_key = "strict_typespan_total_string_F1"
    
    model_directory_val_result_for_sorting = list()
    for model_directory_val_result in model_directory_val_results:
        model_dataset_name, fold, model_fullname, results_val, _, _ = model_directory_val_result
        model_directory_val_result_for_sorting += [(model_dataset_name, fold, model_fullname, results_val[sort_metric_key])]

    model_directory_test_result_for_sorting = list()
    for model_directory_test_result in model_directory_test_results:
        model_dataset_name, fold, model_fullname, results_test, _, _ = model_directory_test_result
        model_directory_test_result_for_sorting += [(model_dataset_name, fold, model_fullname, results_test[sort_metric_key])]

    #Sort based on 4th value (result_test[sort_metric_key]), largest on index 0
    sorted_model_directory_val_results = sorted(model_directory_val_result_for_sorting, key=lambda x: x[3], reverse=True)
    sorted_model_directory_test_results = sorted(model_directory_test_result_for_sorting, key=lambda x: x[3], reverse=True)

    model_dataset_name, fold, _, _ = sorted_model_directory_val_results[0]
    output_filename = f"{model_dataset_name}_crossvalidation_summary.txt"
    summary_filepath = os.path.join(directory_path, output_filename)
    with open(summary_filepath, "w") as output:
        print(f"Writing to {output.name}")
        output.write(f"Best models for {model_dataset_name} based on 'val' gold dataset:\n")
        for i, (model_dataset_name, fold, model_fullname, f1_val) in enumerate(sorted_model_directory_val_results):
            output.write(f"({i+1})   {model_dataset_name}_{fold} Total String F1: {f1_val}\n{model_fullname}\n\n")
        
        output.write("\n\n\n")

        output.write(f"Best models for {model_dataset_name} based on 'test' gold dataset:\n")
        for i, (model_dataset_name, fold, model_fullname, f1_test) in enumerate(sorted_model_directory_test_results):
            output.write(f"({i+1})   {model_dataset_name}_{fold} Total String F1: {f1_test}\n{model_fullname}\n\n")



def _create_models_all_dataframe(model_directory_results, sum_dict, average_dict, variance_dict, std_dict, output_directory, data_type, filtering=False) -> None:
    """
    Creates a pandas dataframe for all results (all models and their checkpoints). The presented averages and std are actually calculated on the top10 results,
    but they should still be a good estimation.
    Saves the dataframe to csv.

    Args:
        model_directory_results (list): List of tuples containing the results for each model.
        sum_dict (dict): Dictionary containing the sum of all results for each metric.
        average_dict (dict): Dictionary containing the average of all results for each metric.
        variance_dict (dict): Dictionary containing the variance of all results for each metric.
        std_dict (dict): Dictionary containing the standard deviation of all results for each metric.
        output_directory (str): Directory to save the dataframe to.
        data_type (str): Either "val" or "test" depending on the data used to calculate the results.
    """
    data_headers = list()
    data_results = list()
    for model_directory_result in model_directory_results:
        model_dataset_name, fold, model_fullname, results, _, _ = model_directory_result

        #If ends on regex "checkpoint_\d+" then extract the checkpoint together with the number
        checkpoint_regex = re.compile(f".*checkpoint-(\d+)")
        match = re.search(checkpoint_regex, model_fullname)
        checkpoint_name = ""
        if match:
            checkpoint_name = "_checkpoint-" + match.group(1)
        column_header = f"{fold}{checkpoint_name}"

        data_headers += [column_header]
        data_results += [results]

    average_header = f"AVERAGE_TOP10_{model_dataset_name}"
    std_header = f"STD_TOP10_{model_dataset_name}"
    variance_header = f"VARIANCE_TOP10_{model_dataset_name}"
    sum_header = f"SUM_TOP10_{model_dataset_name}"

    average_dict = dict(sorted(average_dict.items(), reverse=True))
    std_dict = dict(sorted(std_dict.items(), reverse=True))
    variance_dict = dict(sorted(variance_dict.items(), reverse=True))
    sum_dict = dict(sorted(sum_dict.items(), reverse=True))

    dict_list_prefixed = [average_dict, std_dict, variance_dict, sum_dict] + data_results
    data_headers_prefixed = [average_header, std_header, variance_header, sum_header] + data_headers

    if filtering:
        filtered_list = list()
        for dict_item in dict_list_prefixed:
            filtered_dict = {}
            for key, value in dict_item.items():
                if key in RELEVANT_COMBINATIONS:
                    filtered_dict[key] = value
            filtered_list.append(filtered_dict)
        dict_list_prefixed = filtered_list

    df = pd.DataFrame(dict_list_prefixed).T
    df.columns = data_headers_prefixed
    if filtering:
        output_filepath = os.path.join(output_directory, f"{model_dataset_name}_crossvalidation_dataframe_{data_type}_filtered.csv")
    else:
        output_filepath = os.path.join(output_directory, f"{model_dataset_name}_crossvalidation_dataframe_{data_type}.csv")
    print(f"Writing to Pandas Dataframe to {output_filepath}")
    df_rounded = df.round(4)
    df_rounded.to_csv(output_filepath)



def _write_averages_to_file(average_dict, directory_path, model_full_name, evaluation_type):
    filepath = os.path.join(directory_path, f"{model_full_name}_crossvalidation_{evaluation_type}_average_results.txt")
    with open(filepath, "w") as output:
        for key, value in average_dict.items():
            output.write(f'{key}={value}\n')



def _write_std_to_file(std_dict, directory_path, model_full_name, evaluation_type):
    filepath = os.path.join(directory_path, f"{model_full_name}_crossvalidation_{evaluation_type}_std_results.txt")
    with open(filepath, "w") as output:
        for key, value in std_dict.items():
            output.write(f'{key}={value}\n')



def calculate_average_and_std(fold_results):
    # Extract all keys from fold results
    all_keys = set().union(*fold_results)

    # Calculate the mean dictionary
    sum_dict = {key: 0 for key in all_keys}
    sum_dict = dict(sorted(sum_dict.items()))
    num_dictionaries = len(fold_results)

    for fold in fold_results:
        for key in all_keys:
            sum_dict[key] += fold.get(key, 0)

    average_dict = {key: value / num_dictionaries for key, value in sum_dict.items()}
    average_dict = dict(sorted(average_dict.items()))

    # Calculate the variance dictionary
    variance_dict = {key: 0 for key in all_keys}
    variance_dict = dict(sorted(variance_dict.items()))

    for fold in fold_results:
        for key in all_keys:
            diff = fold.get(key, 0) - average_dict[key]
            variance_dict[key] += diff ** 2

    # Calculate the standard deviation dictionary
    std_dict = {key: (variance / num_dictionaries) ** 0.5 for key, variance in variance_dict.items()}
    std_dict = dict(sorted(std_dict.items()))

    return sum_dict, average_dict, variance_dict, std_dict



def _pick_best_models(
        model_directory_results: List[Tuple[str, str, str, dict, list, list]]
    ) -> Tuple[
            List[Tuple[str, str, str, dict, list, list]],
            List[Tuple[str, str, str, dict, list, list]],
            dict,
            dict
        ]:
    """
    Picks the best model for each fold. This is important because
    often when 1 fold is the best, its checkpoints are better than
    the best model of other folds. Therefore only one model per fold
    is considered in this method.

    Args:
        model_directory_results (list): List of tuples containing the results for each model.

    Returns:
        Tuple[List[Tuple[str, str, str, dict, list, list]], List[Tuple[str, str, str, dict, list, list]], dict, dict]:
            Tuple containing the overall best models and the best models for each fold. For both the corresponding f1 scores.
    """
    overall_best_models = list()
    
    best_models_f1 = {f"fold_{i}": -1. for i in range(0, 10)}
    best_models = {f"fold_{i}": None for i in range(0, 10)}
    total_f1_key = "strict_typespan_total_string_F1"
    for model_directory_result in model_directory_results:
        _, fold_key, _, current_result, _, _ = model_directory_result
        current_f1 = current_result[total_f1_key]

        overall_best_models += [(current_f1, model_directory_result)]

        for fold, f1 in best_models_f1.items():
            if fold_key != fold:
                continue

            if f1 < current_f1:
                best_models_f1[fold] = current_f1
                best_models[fold] = model_directory_result

    #Sort overall_best_models based on the first tuple element i.e. the f1 score
    overall_best_models = sorted(overall_best_models, key=lambda x: x[0], reverse=True)

    overall_best_models_f1 = [x[0] for x in overall_best_models]
    overall_best_models_results = [x[1] for x in overall_best_models]

    return overall_best_models_f1, overall_best_models_results, best_models_f1, best_models



def _pick_best_models(
        model_directory_results: List[Tuple[str, str, str, dict, list, list]]
    ) -> Tuple[
            List[Tuple[str, str, str, dict, list, list]],
            List[Tuple[str, str, str, dict, list, list]],
            dict,
            dict
        ]:
    """
    Picks the best model for each fold. This is important because
    often when 1 fold is the best, its checkpoints are better than
    the best model of other folds. Therefore only one model per fold
    is considered in this method.

    Args:
        model_directory_results (list): List of tuples containing the results for each model.

    Returns:
        Tuple[List[Tuple[str, str, str, dict, list, list]], List[Tuple[str, str, str, dict, list, list]], dict, dict]:
            Tuple containing the overall best models and the best models for each fold. For both the corresponding f1 scores.
    """
    overall_best_models = list()
    
    best_models_f1 = {f"fold_{i}": -1. for i in range(0, 10)}
    best_models = {f"fold_{i}": None for i in range(0, 10)}
    total_f1_key = "strict_typespan_total_string_F1"
    for model_directory_result in model_directory_results:
        _, fold_key, _, current_result, _, _ = model_directory_result
        current_f1 = current_result[total_f1_key]

        overall_best_models += [(current_f1, model_directory_result)]

        for fold, f1 in best_models_f1.items():
            if fold_key != fold:
                continue

            if f1 < current_f1:
                best_models_f1[fold] = current_f1
                best_models[fold] = model_directory_result

    #Sort overall_best_models based on the first tuple element i.e. the f1 score
    overall_best_models = sorted(overall_best_models, key=lambda x: x[0], reverse=True)

    overall_best_models_f1 = [x[0] for x in overall_best_models]
    overall_best_models_results = [x[1] for x in overall_best_models]

    return overall_best_models_f1, overall_best_models_results, best_models_f1, best_models



def flatten_and_sort_results(best_models: Dict[str, List[dict]]) -> List[dict]:
    """
    Extracts the results from the best models dictionary and sorts them.

    Args:
        best_models (Dict[str, List[dict]]): Dictionary containing the best models for each fold.

    Returns:
        List[dict]: List of results for each model.
    """
    fold_results = list()
    for fold, model_information in best_models.items():
        _, _, _, result, _, _ = model_information
        result = dict(sorted(result.items()))
        fold_results += [result]
    return fold_results



def _load_gold_dataset_files(gold_base_dir: str, dataset_directory_name: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns a list with the gold dataset files.

    Args:
        gold_base_dir (str): The base directory where the gold data is stored.
        dataset_directory_name (str): The name of the dataset directory. E.g. pate_single

    Returns:
        Tuple[List[str], List[str], List[str]]: A tuple with the gold directories, gold val filecontents 
                and gold test filecontents.
    """
    gold_directories = [f"{dataset_directory_name}/folds/fold_{i}" for i in range(0, 10)]
    gold_directories = [os.path.join(gold_base_dir, g) for g in gold_directories]
    dataset_name = dataset_directory_name.split("_")[0].strip()

    gold_val_filepaths = list()
    gold_test_filepaths = list()
    for gold_directory in gold_directories:
        gold_val_filepaths.append(os.path.join(gold_directory, f"{dataset_name}-val.bio"))
        gold_test_filepaths.append(os.path.join(gold_directory, f"{dataset_name}-test.bio"))

    gold_val_files = []
    gold_test_files = []
    for val, test in zip(gold_val_filepaths, gold_test_filepaths):
        with open(val, "r") as f:
            gold_val_files.append((val, f.read()))
        with open(test, "r") as f:
            gold_test_files.append((test, f.read()))

    return gold_directories, gold_val_files, gold_test_files



def _get_model_directories(model_base_directory, full_dataset_name):
    target_regex = re.compile(full_dataset_name.upper() + "_" + MODEL_CROSSVALIDATION_FOLDER_SUFFIX_REGEX)
    
    contents = os.listdir(model_base_directory)
    model_toplevel_directories = list()
    for content in contents:
        content_filepath = os.path.join(model_base_directory, content)
        if os.path.isdir(content_filepath):
            if (content.startswith(full_dataset_name.upper()) and re.search(MODEL_CROSSVALIDATION_FOLDER_SUFFIX_REGEX, content) is not None):
                model_toplevel_directories.append(content)
                print(content)
    #sort
    model_toplevel_directories.sort()
    print(model_toplevel_directories)

    model_directories = list()
    for model_toplevel_directory in model_toplevel_directories:
        model_directory_path = os.path.join(model_base_directory, model_toplevel_directory)
        model_timestamps = os.listdir(model_directory_path)
        if len(model_timestamps) > 1:
            print(f"WARNING: More than one timestamp found in {model_directory_path}") 
        for model_timestamp in model_timestamps:
            model_timestamp_directory = os.path.join(model_directory_path, model_timestamp)
            model_directories.append(model_timestamp_directory)
    return model_directories



def _load_model_predictions(model_directory_paths: List[str]):
    val_predictions = list()
    test_predictions = list()
    for model_directory_path in model_directory_paths:
        files = [file for file in os.listdir(model_directory_path) if os.path.isfile(os.path.join(model_directory_path, file))]
        for file in files:
            filepath = os.path.join(model_directory_path, file)
            if file.endswith(VAL_RESULT_FILE_SUFFIX):
                with open(filepath, "r") as f:
                    val_predictions.append((filepath, f.read()))
            elif re.search(TEST_RESULT_FILE_SUFFIX, file) is not None:
                with open(filepath, "r") as f:
                    test_predictions.append((filepath, f.read()))
    return val_predictions, test_predictions



def _generate_negative_cases_analysis(gold_dataset, predict_dataset, negative_cases_indexes):
    """
    Generates the negative cases analysis.

    Args:
        gold_dataset (List[dict]): List of gold entries.
        predict_records (List[Record]): List of predicted records.
        negative_cases_indexes (List[int]): List of indexes of negative cases.

    Returns:
        List[dict]: List of negative cases.
    """
    error_groups = ["total", "time", "date", "set", "duration", "tempexp"]
    evaluation_types = ["strict_typespan", "strict_span", "relaxed_typespan", "relaxed_span"]
    error_classes = [f"{evaluation_type}_{error_group}" for error_group in error_groups for evaluation_type in evaluation_types]
    negative_cases = dict()
    for error_class in error_classes:
        negative_cases_for_class = list()
        for index in negative_cases_indexes[error_class]:
            current_result = dict()
            current_result["error_class"] = error_class
            current_result["index"] = index
            gold_text = gold_dataset[index]["text"]
            current_result["gold_text"] = gold_text
            gold_tokens = gold_dataset[index]["tokens"]
            current_result["gold_tokens"] = gold_tokens
            gold_entity = gold_dataset[index]["entity"]

            gold_entity_offsets = list()
            gold_entity_strings = list()
            for entity in gold_entity:
                entity_type = entity["type"]
                entity_offset_list = tuple(entity["offset"])
                entity_string = entity["text"]
                gold_entity_offsets.append((entity_type, entity_offset_list))
                gold_entity_strings.append((entity_type, entity_string))

            pred_entity = predict_dataset[index]["entity"]
            pred_entity_offsets = list()
            pred_entity_strings = list()
            for entity in pred_entity:
                entity_type = entity["type"]
                entity_offset_list = tuple(entity["offset"])
                entity_string = entity["text"]
                pred_entity_offsets.append((entity_type, entity_offset_list))
                pred_entity_strings.append((entity_type, entity_string))

            is_intersected = False
            for gs in gold_entity_strings:
                gs_type = gs[0]
                gs_text = gs[1]
                for ps in pred_entity_strings:
                    ps_type = ps[0]
                    ps_text = ps[1]
                    if gs_type == ps_type:
                        is_intersected = _has_intersection(gs_text, ps_text)
            current_result["intersection"] = is_intersected
            
            gold_labeled_text = _label_tokens(gold_tokens.copy(), gold_entity_offsets)
            pred_labeled_text = _label_tokens(gold_tokens.copy(), pred_entity_offsets)
            current_result["gold_labeled_text"] = gold_labeled_text
            current_result["pred_labeled_text"] = pred_labeled_text
            current_result["gold_entities"] = list(zip(gold_entity_strings, gold_entity_offsets))
            current_result["pred_entities"] = list(zip(pred_entity_strings, pred_entity_offsets))
            negative_cases_for_class.append(current_result)
        negative_cases[error_class] = negative_cases_for_class
    #Delete empty values for keys 
    negative_cases = {k: v for k, v in negative_cases.items() if v}
    return negative_cases



def _has_intersection(string1, string2):
    """
    Checks if two offsets have an intersection.
    """
    matcher = difflib.SequenceMatcher(None, string1, string2)
    match = matcher.find_longest_match(0, len(string1), 0, len(string2))
    return match.size > 0



def _label_tokens(tokens, offsets):
    token_entity_begins = "["
    token_entity_ends = "]"
    token_type_begins = "<"
    token_type_ends = ">"
    for type, indexes in offsets:
        if len(indexes) == 0:
            continue
        elif len(indexes) == 1:
            start = indexes[0]
            end = indexes[0]
        else:
            start = indexes[0]
            end = indexes[-1]
        
        tokens[start] = f"{token_entity_begins}{token_type_begins}{type}{token_type_ends} {tokens[start]}"
        tokens[end] = f"{tokens[end]}{token_entity_ends}"

    text = ""
    for token in tokens:
        text += token + " "
    return text



def _get_all_model_results(
        gold_val_files: Tuple,
        gold_test_files: Tuple,
        pred_val_files: Tuple,
        pred_test_files: Tuple,
        full_model_name: str,
        debug: bool = False
) -> Tuple[List[Tuple[str, str, str, dict, list, list]], List[Tuple[str, str, str, dict, list, list]]]:
    
    model_directory_val_results = list()
    model_directory_val_negative_cases = list()
    model_directory_test_results = list()
    model_directory_test_negative_cases = list()

    for i, record in enumerate(zip(gold_val_files, pred_val_files, gold_test_files, pred_test_files)):
        fold = f"fold_{i}"
        _, gold_val_file_json = record[0]
        pred_val_filepath, pred_val_file_json = record[1]
        _, gold_test_file_json = record[2]
        pred_test_filepath, pred_test_file_json = record[3]

        def _evaluate(gold_entries, predict_records) -> dict:
            """
            Evaluates the predictions against the gold entries line by line.

            Args:
                gold_entries (List[dict]): List of gold entries.
                predict_records (List[Record]): List of predicted records.

            Returns:
                dict: Dictionary containing the results of the evaluation.
            """
            scorer = TemporalTypeScorer()

            #Format gold and pred to match in structure
            gold_instance_list = [x["entity"] for x in gold_entries]
            pred_instance_list = [x["entity"] for x in predict_records]

            gold_instance_list_restructured = scorer.load_gold_list(gold_instance_list)
            pred_instance_list_restructured = scorer.load_gold_list(pred_instance_list)

            sub_results, negative_case_indexes = scorer.eval_instance_list(
                gold_instance_list=gold_instance_list_restructured,
                pred_instance_list=pred_instance_list_restructured
            )
            return sub_results, negative_case_indexes


        val_results, negative_val_cases_indexes = _evaluate(gold_val_file_json, pred_val_file_json)
        test_results, negative_test_cases_indexes = _evaluate(gold_test_file_json, pred_test_file_json)
        
        val_negative_cases = _generate_negative_cases_analysis(gold_val_file_json, pred_val_file_json, negative_val_cases_indexes)
        test_negative_cases = _generate_negative_cases_analysis(gold_test_file_json, pred_test_file_json, negative_test_cases_indexes)

        model_val_results = (full_model_name, fold, pred_val_filepath, val_results, pred_val_file_json, None)
        model_test_results = (full_model_name, fold, pred_test_filepath, test_results, pred_test_file_json, None)
        
        if debug:
            print()
            print(f"Results for model {full_model_name}_{fold} (val):")
            pprint.pprint(model_val_results[3])
            print()
            print()
            print(f"Results for model {full_model_name}_{fold} (test):")
            pprint.pprint(model_test_results[3])
            print("\n" + "-" * 75)
        model_directory_val_results += [model_val_results]
        model_directory_test_results += [model_test_results]
        model_directory_val_negative_cases += [val_negative_cases]
        model_directory_test_negative_cases += [test_negative_cases]

    return model_directory_val_results, model_directory_test_results, model_directory_val_negative_cases, model_directory_test_negative_cases
    


def _bio_to_json(bio: str) -> List[Dict[str, Any]]:
    """
    Converts a BIO string to a list of dictionaries.

    Args:
        bio (str): The BIO string.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries.
    """
    biotag_to_tag = {
        "B-DATE": "date",
        "I-DATE": "date",
        "B-DURATION": "duration",
        "I-DURATION": "duration",
        "B-SET": "set",
        "I-SET": "set",
        "B-TIME": "time",
        "I-TIME": "time",
        "B-TEMPEXP": "tempexp",
        "I-TEMPEXP": "tempexp",
        "O": "O"
    }
    bio = bio.strip("\n\t" + " ")
    bio = bio.split("\n\n")
    jsonlist = list()
    for sentence in bio:
        sentence_json = dict()
        pairs = sentence.split("\n")
        tokens = [x.split("\t")[0] for x in pairs]
        tags = [x.split("\t")[1] for x in pairs]
        sentence_json["tokens"] = tokens
        sentence_json["text"] = " ".join(tokens).strip()

        #Extract entities
        entities = list()
        entity = None
        for i, tag in enumerate(tags):
            lower_tag = tag.lower()
            if lower_tag.startswith("b-"):
                entity = dict()
                entity["type"] = biotag_to_tag[tag]
                entity["offset"] = [i]
                entity["text"] = tokens[i]
            elif lower_tag.startswith("i-"):
                if entity is not None:
                    entity["offset"].append(i)
                    entity["text"] += " " + tokens[i]
                    entity["text"] = entity["text"].strip()
                else:
                    raise ValueError(f"Found I- tag without B- tag: {tag}")
            elif lower_tag.startswith("o"):
                if entity is not None:
                    entities.append(entity)
                    entity = None
            else:
                raise ValueError(f"Unknown tag {tag}")
            
            if i == len(tags) - 1:
                if entity is not None:
                    entities.append(entity)
            
        sentence_json["entity"] = entities
        jsonlist.append(sentence_json)
        #print(json.dumps(sentence_json, indent=4) + "\n" * 3)
    return jsonlist



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", "-bm", type=str, default="logs")
    parser.add_argument("--base_data_dir", "-bd", type=str, default="../temporal-data/entity/my_converted_datasets/bio")
    parser.add_argument("--dataset_name", "-d", type=str) #pate, snips, tempeval, ...
    parser.add_argument("--classes", "-c", type=str) #multi or single
    parser.add_argument("--copy_gold_files", "-g", action="store_true")
    parser.add_argument("--output_base_dir", "-o", type=str, default="crossvalidation-output")
    args = parser.parse_args()

    gold_base_directory = args.base_data_dir
    model_base_directory = args.base_model_dir
    dataset_name = args.dataset_name
    classes = args.classes
    full_dataset_name = f"{args.dataset_name}_{args.classes}" #e.g. "pate_multi"
    copy_gold_files = args.copy_gold_files
    logfiles_output_directory = f"{args.output_base_dir}/{full_dataset_name}_crossvalidation_logfiles"

    full_dataset_name = f"{dataset_name}_{classes}"
    print("Initializing crossvalidation...")
    print(f"Dataset name: {dataset_name}")
    print(f"Model base directory: {model_base_directory}")
    print(f"Data base directory: {gold_base_directory}")
    print(f"Full dataset name: {full_dataset_name}")
    print(f"Classes: {classes}")
    print(f"Copy gold files: {copy_gold_files}")


    #Grab the gold and pred data
    _, gold_val_files_bio, gold_test_files_bio = _load_gold_dataset_files(gold_base_directory, full_dataset_name)
    model_directories = _get_model_directories(model_base_directory, full_dataset_name)
    val_predictions_bio, test_predictions_bio = _load_model_predictions(model_directories)

    #Convert to json
    val_gold_json = [(x[0], _bio_to_json(x[1])) for x in gold_val_files_bio]
    test_gold_json = [(x[0], _bio_to_json(x[1])) for x in gold_test_files_bio]
    val_predictions_json = [(x[0], _bio_to_json(x[1])) for x in val_predictions_bio]
    test_predictions_json = [(x[0], _bio_to_json(x[1])) for x in test_predictions_bio]

    #Check correspondence
    for i, record in enumerate(zip(val_gold_json, val_predictions_json, test_gold_json, test_predictions_json)):
        if (
            not (f"fold_{i}" in record[0][0].lower() 
            and f"fold_{i}" in record[1][0].lower() 
            and f"fold_{i}" in record[2][0].lower()
            and f"fold_{i}" in record[3][0].lower())
        ):
            raise ValueError(f"Bad sorting of gold and pred files found in fold {i}")

    #Analyse the data
    model_directory_val_results, model_directory_test_results, \
    model_directory_val_negative_cases, model_directory_test_negative_cases  = _get_all_model_results(
        gold_val_files=val_gold_json,
        gold_test_files=test_gold_json,
        pred_val_files=val_predictions_json,
        pred_test_files=test_predictions_json,
        full_model_name=full_dataset_name,
        debug=True
    )

    if not os.path.exists(logfiles_output_directory):
        os.makedirs(logfiles_output_directory)


    """
    Analyze results of "val" and generate report files
    """
    overall_best_model_f1_val, overall_best_model_val, best_models_f1_val, best_models_val = _pick_best_models(model_directory_val_results)
    fold_results_val = flatten_and_sort_results(best_models_val) #best_models_val is cruical
    sum_dict_val, average_dict_val, variance_dict_val, std_dict_val = calculate_average_and_std(fold_results_val)
    _write_averages_to_file(average_dict_val, logfiles_output_directory, full_dataset_name, "val")
    _write_std_to_file(std_dict_val, logfiles_output_directory, full_dataset_name, "val")
    _create_models_all_dataframe(
        model_directory_results = model_directory_val_results, 
        sum_dict = sum_dict_val, 
        average_dict = average_dict_val, 
        variance_dict = variance_dict_val, 
        std_dict = std_dict_val, 
        output_directory = logfiles_output_directory, 
        data_type = "val-all",
        filtering=True
    )

    _create_models_all_dataframe(
        model_directory_results = model_directory_val_results, 
        sum_dict = sum_dict_val, 
        average_dict = average_dict_val, 
        variance_dict = variance_dict_val, 
        std_dict = std_dict_val, 
        output_directory = logfiles_output_directory, 
        data_type = "val-all",
        filtering=False
    )
    _create_models_best_dataframe(best_models_val, sum_dict_val, average_dict_val, variance_dict_val, std_dict_val, logfiles_output_directory, "val-best", filtering=True)
    _create_models_best_dataframe(best_models_val, sum_dict_val, average_dict_val, variance_dict_val, std_dict_val, logfiles_output_directory, "val-best", filtering=False)

    _write_model_results_to_files(
        model_directory_results = model_directory_val_results, 
        directory_path = logfiles_output_directory,
        evaluation_type = "val"
    )

    _write_error_analysis_to_files(
        directory_path = logfiles_output_directory,
        evaluation_type = "val",
        negative_cases = model_directory_val_negative_cases,
        results = model_directory_val_results
    )
    

    """
    Analyze results of "test" and generate report files
    """
    overall_best_model_f1_test, overall_best_model_test, best_models_f1_test, best_models_test = _pick_best_models(model_directory_test_results)
    fold_results_test = flatten_and_sort_results(best_models_test) #best_models_test is cruical
    sum_dict_test, average_dict_test, variance_dict_test, std_dict_test = calculate_average_and_std(fold_results_test)
    _write_averages_to_file(average_dict_test, logfiles_output_directory, full_dataset_name, "test")
    _write_std_to_file(std_dict_test, logfiles_output_directory, full_dataset_name, "test")
    _create_models_all_dataframe(model_directory_test_results, sum_dict_test, average_dict_test, variance_dict_test, std_dict_test, logfiles_output_directory, "test-all", filtering=True)
    _create_models_all_dataframe(model_directory_test_results, sum_dict_test, average_dict_test, variance_dict_test, std_dict_test, logfiles_output_directory, "test-all", filtering=False)
    _create_models_best_dataframe(best_models_test, sum_dict_test, average_dict_test, variance_dict_test, std_dict_test, logfiles_output_directory, "test-best", filtering=True)
    _create_models_best_dataframe(best_models_test, sum_dict_test, average_dict_test, variance_dict_test, std_dict_test, logfiles_output_directory, "test-best", filtering=False)

    _write_model_results_to_files(
        model_directory_results = model_directory_test_results, 
        directory_path = logfiles_output_directory,
        evaluation_type = "test"
    )

    _write_error_analysis_to_files(
        directory_path = logfiles_output_directory,
        evaluation_type = "test",
        negative_cases = model_directory_test_negative_cases,
        results = model_directory_test_results
    )

    _write_f1_summary_to_file(
        model_directory_val_results = model_directory_val_results,
        model_directory_test_results = model_directory_test_results,
        directory_path = logfiles_output_directory
    )

    _create_models_structured_dataframe(model_directory_val_results, average_dict_val, std_dict_val, logfiles_output_directory, "val")
    _create_models_structured_dataframe(model_directory_test_results, average_dict_test, std_dict_test, logfiles_output_directory, "test")

    
    """
    General files
    """
    _write_best_models_to_file(
        directory_path = logfiles_output_directory, 
        model_full_name = full_dataset_name, 
        best_models_val = overall_best_model_val, 
        best_models_f1_val = overall_best_model_f1_val, 
        best_models_test = overall_best_model_test, 
        best_models_f1_test = overall_best_model_f1_test
    )

    _write_best_fold_models_to_file(
        directory_path = logfiles_output_directory, 
        model_full_name = full_dataset_name, 
        best_models_val = best_models_val, 
        best_models_f1_val = best_models_f1_val, 
        best_models_test = best_models_test, 
        best_models_f1_test = best_models_f1_test
    )


    """
    Copy gold files to logfiles directory for convenience 
    """
    if copy_gold_files:
        gold_val_files_paths = [x[0] for x in gold_val_files_bio]
        gold_test_files_paths = [x[0] for x in gold_test_files_bio]

        _copy_gold_files(
            gold_val_files = gold_val_files_paths,
            gold_test_files = gold_test_files_paths, 
            output_directory = logfiles_output_directory,
            model_full_name = full_dataset_name
        )
    

if __name__ == '__main__':
    main()