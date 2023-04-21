"""
Helper functions.
@author: Yun-Shiuan Chuang
@contact: yunshiuan.chuang@gmail.com
"""

from torch import nn
import torch as th
import numpy as np
from datasets.load import load_metric
from parameters_meta import ParametersMeta as par_meta
from parameters_semeval import ParametersSemEval2016 as par_sem_eval
import pandas as pd
import re
from sklearn.utils import resample
from datasets.arrow_dataset import Dataset
from sklearn.metrics import confusion_matrix
import os
from os import path, makedirs
from os.path import join
from itertools import chain
from glob import glob
from shutil import rmtree
from sklearn.model_selection import train_test_split
import yaml
from transformers import AutoConfig
from datetime import datetime
from collections import Counter
from PIL import Image as PILImage
from IPython.display import display

def get_parameters_for_dataset(dataset_meta=None):
    if dataset_meta is None:
        dataset_meta = par_meta.DATASET_META
    assert dataset_meta in par_meta.DATASET_META

    if dataset_meta == "SEM_EVAL":
        par = par_sem_eval
    else:
        raise ValueError("Invalid dataset_meta: {}".format(dataset_meta))
    return par


par = get_parameters_for_dataset()


def process_dataframe(input_csv, dataset, input_mode="csv", input_df=None, keep_ts=False, list_col_of_interest_input=None, list_col_of_interest_output=None):
    """Process the input csv file and return a dataframe.
    Args:
        input_csv (str): the path to the input csv file.
        dataset (str): the dataset. Must be one of the following:
            - COVID_VACCINE_Q1
            - COVID_VACCINE_Q2
        input_mode (str): the input mode. Must be one of the following:
            - csv: the input is a csv file.
            - df: the input is a dataframe.
        input_df (pd.DataFrame): the input dataframe. Only used if `input_mode` is "df".
        keep_ts (bool): whether to keep the timestamp. Default: False.
        list_col_of_interest_input (list): the list of columns of interest in the input dataframe. Default: None.
        list_col_of_interest_output (list): the list of columns of interest in the output dataframe. Default: None.

    Returns:
        df (pd.DataFrame): the processed dataframe.
    """
    if input_mode == "csv":
        assert path.isfile(input_csv), "input_csv does not exist: {}".format(input_csv)
        assert input_df is None, "input_df must be None if input_mode is csv"
    elif input_mode == "df":
        assert input_csv is None, "input_csv must be None if input_mode is df"
        assert isinstance(input_df, pd.DataFrame), "input_df must be a pd.DataFrame"
    # make sure to read tweet_id as str
    assert ("COVID_VACCINE" in dataset) or (dataset == "WTWT") or (dataset == "SEM_EVAL"), "dataset must be COVID_VACCINE_* or WTWT"
    if "COVID_VACCINE" in dataset:
        par = get_parameters_for_dataset("COVID_VACCINE")
        if input_mode == "csv":
            df = pd.read_csv(input_csv, header=[0], dtype={"tweet_id": str})
        else:
            df = input_df
        if list_col_of_interest_input is None:
            list_col_of_interest_input = ["tweet", "month", "label_majority", "tweet_id"]
        if list_col_of_interest_output is None:
            list_col_of_interest_output = ["text", "month", "label", "tweet_id"]

    elif dataset == "WTWT":
        par = get_parameters_for_dataset("WTWT")
        if input_mode == "csv":
            df = pd.read_csv(input_csv, header=[0], dtype={"row_id": str})
        else:
            df = input_df
        if list_col_of_interest_input is None:
            list_col_of_interest_input = ["tweet", "month", "label", "row_id"]
        if list_col_of_interest_output is None:
            list_col_of_interest_output = ["text", "month", "label", "row_id"]
    elif dataset == "SEM_EVAL":
        par = get_parameters_for_dataset("SEM_EVAL")
        if input_mode == "csv":
            df = pd.read_csv(input_csv, header=[0], dtype={"ID": str})
        else:
            df = input_df
        if list_col_of_interest_input is None:
            list_col_of_interest_input = ["tweet", "label", "ID"]
        if list_col_of_interest_output is None:
            list_col_of_interest_output = ["text", "label", "ID"]

    if keep_ts:
        list_col_of_interest_input.append("created_at")
        list_col_of_interest_output.append("timestamp")
    df = df[list_col_of_interest_input]
    df.columns = pd.Series(list_col_of_interest_output)
    # should have no NA
    assert df.isna().sum().sum() == 0
    # recode the label into number
    df['label'] = df['label'].map(par.DICT_STANCES_CODE[dataset])
    return df


def preprocess_dataset(df, tokenizer, col_name_text="text", with_label=True, keep_tweet_id=False, keep_ts=False,
                       name_model=None, output_batched=False, output_batch_size=16, pad_to_max_in_dataset=False, col_name_tweet_id="tweet_id"):
    """Preprocess the dataset for HuggingFace.
    Args:
        df (pd.DataFrame): the input dataframe.
        tokenizer (PreTrainedTokenizer): the tokenizer.
        col_name_text (str): the name of the column that contains the texts.
        with_label (bool): whether the output dataset should contain labels.
        keep_tweet_id (bool): whether to keep the tweet_id. Default: False.
        keep_ts (bool): whether to keep the timestamp. Default: False.
        name_model (str): the type of the model. If None, the default model type is used.
        output_batched (bool): whether the output dataset should be batched. This is useful if one wants to directly iterate over the dataset without using the Trainer, e.g., as in FlanT5XxlLabelPredictor.predict_labels().
            - see: https://discuss.huggingface.co/t/streaming-batched-data/21603
        output_batch_size (int): the batch size of the output dataset if `output_batched` is True.
        pad_to_max_in_dataset (bool): whether to pad the input to the max length in the entire dataset. This avoid any truncation in the inputs. Default: False.
        col_name_tweet_id (str): the name of the column that contains the tweet_id. Default: "tweet_id".
    Returns:
        dataset (Dataset): the preprocessed dataset.
    """
    assert name_model in [None, "flan-t5-xxl", "flan-ul2", "flan-t5-small", "flan-t5-large"], "model_type not supported"
    list_col_of_interest_input = [col_name_text]
    if keep_ts:
        list_col_of_interest_input.append("timestamp")
    if keep_tweet_id:
        list_col_of_interest_input.append(col_name_tweet_id)
    if with_label:
        list_col_of_interest_input.append("label")

    df = df[list_col_of_interest_input]

    dataset = Dataset.from_pandas(df)

    # pad to the longest sequence in the entire dataset
    if pad_to_max_in_dataset:
        # get the longest seq in each batch
        # - https://huggingface.co/docs/transformers/pad_truncation
        dataset_find_max = dataset.map(lambda e: tokenizer(e[col_name_text], padding='longest', return_tensors="pt"), batched=True)
        max_length_dataset = max(len(x) for x in dataset_find_max['input_ids'])
        # truncate to the max length in the dataset
        dataset = dataset.map(lambda e: tokenizer(e[col_name_text], truncation=True, padding='max_length', max_length=max_length_dataset), batched=True)
    else:
        if name_model in ["flan-t5-xxl", "flan-ul2", "flan-t5-small", "flan-t5-large"]:
            # pad to the longest sequence in the batch to avoid truncation
            # - https://huggingface.co/docs/transformers/pad_truncation
            dataset = dataset.map(lambda e: tokenizer(e[col_name_text], padding='longest', return_tensors="pt"), batched=True)
        else:
            dataset = dataset.map(lambda e: tokenizer(e[col_name_text], truncation=True, padding='max_length', max_length=256), batched=True)
    # max_length=512: the default max length of "bert-base-cased"

    # None: e.g., bert
    if name_model is None:
        cols_tokenized_output = [col_name_text] + ['input_ids', 'token_type_ids', 'attention_mask']
    elif name_model in ["flan-t5-xxl", "flan-ul2", "flan-t5-small", "flan-t5-large"]:
        cols_tokenized_output = [col_name_text] + ['input_ids', 'attention_mask']

    if keep_ts:
        cols_tokenized_output.append('timestamp')
    if keep_tweet_id:
        cols_tokenized_output.append(col_name_tweet_id)
    if with_label:
        cols_tokenized_output.append('label')

    dataset.set_format(type='torch', columns=cols_tokenized_output)

    if output_batched:
        def group_batch(batch):
            return {k: [v] for k, v in batch.items()}
        # https://discuss.huggingface.co/t/streaming-batched-data/21603
        dataset = dataset.map(group_batch, batched=True, batch_size=output_batch_size)
    return dataset


def partition_and_resample_df(df, seed, partition_type, list_time_domain=None, list_source_domain=None, factor_upsample=0, do_downsample=False,
                              partition_source_method=["train", "vali"], partition_target_method="all", read_partition_from_df=False, df_partitions=None, col_name_label="label"):
    """Partition the dataframe into
        - single_domain
            - training, validation, and test sets.
        - gda (gradual domain adaptation)
            - source domain: training and validation sets.
            - target domains
        - pseudo-labels
            - training, validation
            - note that resampling is performed on the pseudo-labels rather than the true labels.

    Args:
        df (pd.DataFrame): the input dataframe.
        seed (int): the random seed.
        partition_type (str): the type of partition. Must be one of the following:
            - single_domain
            - gda
            - pseudo-labels
        list_time_domain: the list of time domains to be used when partition_type is "gda".
        list_source_domain: the list of source domains to be used when partition_type is "gda".
        factor_upsample (int): the factor of upsampling. How many times the majority class is upsampled.
            - 0 if no upsampling
            - 1 if the majority class it not upsampled, and the other minority classes are upsampled to the same size as the majority class
            - 2 if the majority class is upsampled twice, and the other minority classes are upsampled to twice the same size of the majority class
        do_downsample: whether to downsample such that the number of samples in each class is the same, matching the size to the minority class.
        partition_source_method (list(str)): Used only if `partition_type == "gda"`. The method to partition the source domain. Must be one of the following:
            - ["train","vali"]: partition the source domain into training and validation sets.
            - ["train","vali","test"]: partition the source domain into training, validation, and test sets. This is useful if one wants to compare gda against single-domain or GPT in the source domain.
            Default: ["train","vali"].
        partition_target_method (list(str)): Used only if `partition_type == "gda"`. The method to partition the target domain. Must be one of the following:
            - ["all"]: use all the data in the target domain.
            - ["all","vali","test"]: Include "all" and "vali", "test" partitions (both are subsets of "all").  This is useful if one wants to compare gda against single-domain or GPT in the target domains because the test partition is not used in the training in "single-domain".
            Default: ["all"].
        read_partition_from_df (bool): whether to read the partition from the dataframe. This is useful when the partition is already saved in the dataframe.
        df_partitions (pd.DataFrame): the dataframe containing the partition. This is useful when the partition is already saved in the dataframe.
        col_name_label (str): the name of the column containing the label. Default: "label".

    Returns:
        dict_partition (dict): the partitioned and resampled dataframe.
            - single_domain: e.g., {train: df, vali: df, test: df}
            - gda: e.g., {source_train: df, source_vali: df, target_2022-10: df, target_2022-11: df, ...}
            - pseudo-labels, e.g., {train: df, vali: df}
    """
    par = get_parameters_for_dataset()
    assert partition_type in ["single_domain", "gda", "pseudo-labels"], "partition_type must be one of the following: single_domain, gda, pseudo-labels"
    assert not (do_downsample and factor_upsample != 0), "Cannot upsample and downsample at the same time!"
    assert not (read_partition_from_df and df_partitions is None), "df_partitions must be specified when read_partition_from_df is True"

    if partition_type == "gda":
        assert list_time_domain is not None, "list_time_domain must be specified when partition_type is gda"
        assert len(list_time_domain) > 0, "list_time_domain cannot be empty"
        assert list_source_domain is not None, "list_source_domain must be specified when partition_type is gda"
        assert set(list_source_domain).issubset(set(list_time_domain)), "list_source_domain must be a subset of list_time_domain"
        assert len(list_source_domain) > 0, "list_source_domain cannot be empty"
        assert partition_source_method in [["train", "vali"], ["train", "vali", "test"]], "partition_source_method must be one of the following: ['train','vali'], ['train','vali','test']"
        assert partition_target_method in [["all"], ["all", "vali", "test"]], "partition_target_method must be one of the following: ['all'], ['all','vali','test']"

    if partition_type == "single_domain":
        # {train: df, vali: df, test: df}
        dict_df = dict()

        # partition the source domain into train and vali
        if not read_partition_from_df:
            df_train, df_vali_test = train_test_split(df,
                                                      test_size=par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["vali"] + par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["test"],
                                                      random_state=seed)
            df_vali, df_test = train_test_split(df_vali_test,
                                                test_size=par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["test"] /
                                                (par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["vali"] + par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["test"]),
                                                random_state=seed)
        else:
            # read the partition from the dataframe
            # - join by par.TEXT_ID
            df_train = df.merge(df_partitions[df_partitions["partition"] == "train"][[par.TEXT_ID]],
                                on=par.TEXT_ID, how="inner")
            df_vali = df.merge(df_partitions[df_partitions["partition"] == "vali"][[par.TEXT_ID]],
                               on=par.TEXT_ID, how="inner")
            df_test = df.merge(df_partitions[df_partitions["partition"] == "test"][[par.TEXT_ID]],
                               on=par.TEXT_ID, how="inner")

        dict_df["train_raw"] = df_train
        dict_df["vali_raw"] = df_vali
        dict_df["test_raw"] = df_test

        # ----------
        # Upsample the minority class in 'train', 'vali', and 'test'
        # - should preserve the raw unbalanced set for evaluation purpose
        # - raw unbalanced sets: train, vali, test
        # - upsampled sets: train_upsampled, vali_upsampled, test_upsampled
        # ----------
        if factor_upsample > 0:
            for data_set in ["train_raw", "vali_raw", "test_raw"]:
                df_upsampled = upsample_df(dict_df[data_set], col_name_label, factor_upsample, seed)
                key_upsampled = re.sub("_raw", "_upsampled", data_set)
                dict_df[key_upsampled] = df_upsampled

        # ----------
        # Downsample the majority class in 'train', 'vlai', and 'test'
        # - should preserve the raw unbalanced set for evaluation purpose
        # - raw unbalanced sets: train, vali, test
        # - downsampled sets: train_downsampled, vali_downsampled, test_downsampled
        # ----------
        if do_downsample:
            for data_set in ["train_raw", "vali_raw", "test_raw"]:
                df_downsampled = downsample_df(dict_df[data_set], col_name_label, seed)
                key_downsampled = re.sub("_raw", "_downsampled", data_set)
                dict_df[key_downsampled] = df_downsampled
        return dict_df

    elif partition_type == "gda":
        # ----------
        # Partition the data into domains
        # - source train
        # - source vali
        # - target 1,2,3,...
        # ----------
        # {source_train: df, source_vali: df, target_2022-10: df, target_2022-11: df, ...}
        dict_df = dict()
        list_df_source = []
        # - save the time domains for later use
        list_source_time_domain = []
        list_target_time_domain = []
        for time_domain in list_time_domain:
            name_time_domain = convert_time_unit_into_name(time_domain)
            # source domains
            if time_domain in list_source_domain:
                list_df_source.append(df[df.month.isin(time_domain)])
                list_source_time_domain.append(time_domain)
            else:
                # target domains
                dict_df["target_" + name_time_domain + "_all"] = df[df.month.isin(time_domain)]
                list_target_time_domain.append(time_domain)

        # merge the source domain acorss time units
        df_source = pd.concat(list_df_source)

        # ----------
        # partition the source domain
        # ----------
        if not read_partition_from_df:
            if partition_source_method == ["train", "vali"]:
                df_source_train, df_source_vali = train_test_split(df_source,
                                                                   test_size=par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["vali"] +
                                                                   par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["test"],
                                                                   random_state=seed)
            elif partition_source_method == ["train", "vali", "test"]:
                df_source_train, df_source_vali_test = train_test_split(df_source,
                                                                        test_size=par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["vali"] +
                                                                        par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["test"],
                                                                        random_state=seed)
                df_source_vali, df_source_test = train_test_split(df_source_vali_test,
                                                                  test_size=par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["test"] /
                                                                  (par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["vali"] + par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["test"]),
                                                                  random_state=seed)

        else:
            # read the partition from the dataframe
            # - join by par.TEXT_ID
            df_source_train = df_source.merge(df_partitions[df_partitions["partition"] == "train"][[par.TEXT_ID]], on=par.TEXT_ID, how="inner")
            if partition_source_method == ["train", "vali"]:
                df_source_vali = df_source.merge(df_partitions[df_partitions["partition"].isin(["vali", "test"])][[par.TEXT_ID]], on=par.TEXT_ID, how="inner")
            if partition_source_method == ["train", "vali", "test"]:
                df_source_vali = df_source.merge(df_partitions[df_partitions["partition"] == "vali"][[par.TEXT_ID]], on=par.TEXT_ID, how="inner")
                df_source_test = df_source.merge(df_partitions[df_partitions["partition"] == "test"][[par.TEXT_ID]], on=par.TEXT_ID, how="inner")

        dict_df["source_train_raw"] = df_source_train
        dict_df["source_vali_raw"] = df_source_vali
        if partition_source_method == ["train", "vali", "test"]:
            dict_df["source_test_raw"] = df_source_test

        # ----------
        # partition the target domains
        # ----------
        if partition_target_method == ["all"]:
            pass
        elif partition_target_method == ["all", "vali", "test"]:
            # collect the partitioned target domains
            dict_target_partitions = dict()
            for time_unit in dict_df:
                if time_unit.startswith("target"):
                    if not read_partition_from_df:
                        _, df_target_vali_test = train_test_split(dict_df[time_unit],
                                                                  test_size=par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["vali"] +
                                                                  par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["test"],
                                                                  random_state=seed)
                        df_target_vali, df_target_test = train_test_split(df_target_vali_test,
                                                                          test_size=par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["test"] /
                                                                          (par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["vali"] +
                                                                           par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["test"]),
                                                                          random_state=seed)
                        # replace "all" by "vali" and "test"
                        time_unit_vali = re.sub("_all", "_vali", time_unit)
                        time_unit_test = re.sub("_all", "_test", time_unit)
                        # collect the partitioned target domains
                        dict_target_partitions[time_unit_vali] = df_target_vali
                        dict_target_partitions[time_unit_test] = df_target_test
                    else:
                        # read the partition from the dataframe
                        # - join by par.TEXT_ID
                        df_target_vali = dict_df[time_unit].merge(df_partitions[df_partitions["partition"].isin(["vali"])][[par.TEXT_ID]], on=par.TEXT_ID, how="inner")
                        df_target_test = dict_df[time_unit].merge(df_partitions[df_partitions["partition"] == "test"][[par.TEXT_ID]], on=par.TEXT_ID, how="inner")
                        # replace "all" by "vali" and "test"
                        time_unit_vali = re.sub("_all", "_vali", time_unit)
                        time_unit_test = re.sub("_all", "_test", time_unit)
                        # collect the partitioned target domains
                        dict_target_partitions[time_unit_vali] = df_target_vali
                        dict_target_partitions[time_unit_test] = df_target_test
            # merge the two dictionaries
            dict_df = {**dict_df, **dict_target_partitions}

        # ----------
        # Upsample the minority class in 'source_train'
        # - should preserve the raw unbalanced set for evaluation purpose
        # - raw unbalanced sets: train, vali, {test...}
        # - upsampled sets: train_upsampled, vali_upsampled, {test...}_upsampled
        # ----------
        if partition_source_method == ["train", "vali"]:
            list_source_partitions = ["source_train_raw", "source_vali_raw"]
        elif partition_source_method == ["train", "vali", "test"]:
            list_source_partitions = ["source_train_raw", "source_vali_raw", "source_test_raw"]

        if factor_upsample > 0:
            for data_set in list_source_partitions:
                df_upsampled = upsample_df(dict_df[data_set], col_name_label, factor_upsample, seed)
                key_upsampled = re.sub("_raw", "_upsampled", data_set)
                dict_df[key_upsampled] = df_upsampled

        # ----------
        # Downsample the majority class in 'train', 'vlai', and 'test'
        # - should preserve the raw unbalanced set for evaluation purpose
        # - raw unbalanced sets: train, vali, test
        # - downsampled sets: train_downsampled, vali_downsampled, test_downsampled
        # ----------
        if do_downsample:
            for data_set in list_source_partitions:
                df_downsampled = downsample_df(dict_df[data_set], col_name_label, seed)
                key_downsampled = re.sub("_raw", "_downsampled", data_set)
                dict_df[key_downsampled] = df_downsampled
        return dict_df

    elif partition_type == "pseudo-labels":
        # {train: df, vali: df}
        dict_df = dict()

        if not read_partition_from_df:
            # partition the source domain into train and vali
            # - this may include test data due to the random split
            # -- test_size is vali+test due to legacy reasons (before data version 20230321)
            df_train, df_vali = train_test_split(df,
                                                 test_size=par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["vali"] + par.DICT_RATIO_PARTITION_TRAIN_VALI_TEST["test"],
                                                 random_state=seed)
        else:
            # read the partition from the dataframe
            # - join by par.TEXT_ID
            df_train = df.merge(df_partitions[df_partitions["partition"].isin(["train"])][[par.TEXT_ID]], on=par.TEXT_ID, how="inner")
            # - avoid using the test set during training at all
            df_vali = df.merge(df_partitions[df_partitions["partition"].isin(["vali"])][[par.TEXT_ID]], on=par.TEXT_ID, how="inner")
        dict_df["train_raw"] = df_train
        dict_df["vali_raw"] = df_vali

        # ----------
        # Upsample the minority class in 'train', 'vali', and 'test'
        # - should preserve the raw unbalanced set for evaluation purpose
        # - raw unbalanced sets: train, vali, test
        # - upsampled sets: train_upsampled, vali_upsampled, test_upsampled
        # ----------
        if factor_upsample > 0:
            for data_set in ["train_raw", "vali_raw"]:
                df_upsampled = upsample_df(dict_df[data_set], 'predicted_label', factor_upsample, seed)
                key_upsampled = re.sub("_raw", "_upsampled", data_set)
                dict_df[key_upsampled] = df_upsampled

        # ----------
        # Downsample the majority class in 'train', 'vlai', and 'test'
        # - should preserve the raw unbalanced set for evaluation purpose
        # - raw unbalanced sets: train, vali, test
        # - downsampled sets: train_downsampled, vali_downsampled, test_downsampled
        # ----------
        if do_downsample:
            for data_set in ["train_raw", "vali_raw"]:
                df_downsampled = downsample_df(dict_df[data_set], 'predicted_label', seed)
                key_downsampled = re.sub("_raw", "_downsampled", data_set)
                dict_df[key_downsampled] = df_downsampled
        return dict_df


def initialize_llm_config(config, llm_type):
    """Initialize the config for LLM.

    Args:
        config (dict): the config
        llm_type (str): the type of LLM, e.g., "bert"
    Returns:
        llm_config (dict): the config for LLM
    """
    assert llm_type in ["bert"]

    if llm_type == "bert":
        # - customize the config for bert
        # -- https://stackoverflow.com/questions/64947064/transformers-pretrained-model-with-dropout-setting
        llm_config = AutoConfig.from_pretrained(config.encoder)
        llm_config.hidden_dropout_prob = config.final_layer_dropout_rate
        llm_config.attention_probs_dropout_prob = config.bert_attention_dropout_rate
        llm_config.num_labels = config.num_labels
    return llm_config


def freeze_llm(model, llm_type):
    """Freeze the LLM.

    Args:
        model (nn.Module): the model
        llm_type (str): the type of LLM, e.g., "bert"
    """
    assert llm_type in ["bert"]

    if llm_type == "bert":
        for param in model.base_model.parameters():
            param.requires_grad = False


def reinitialize_bert_top_layers(bert, num_layers):
    """Reinitialize the top layers of BERT.
    Args:
        bert (BertModel): the BERT model
        num_layers (int): the top-N layers to reinitialize. The top-1 layer is the pooler layer, and the rest is the decoder.
    Note:
        - for how weights in BERT are initialized, see:
            - https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bert/modeling_bert.py#L748
    """
    assert num_layers >= 0, "num_layers must be greater than or equal to 0."
    WEIGHTS_STD = 0.02

    if num_layers == 0:
        # no need to reinitialize
        return
    for i_layer in range(num_layers):
        if i_layer == 0:
            # ----------
            # reinitialize the pooler layer
            # - https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bert/modeling_bert.py#L873
            # - https://github.com/pytorch/pytorch/pull/5617#issuecomment-371435360
            # ----------
            bert.pooler.dense.weight.data.normal_(mean=0.0, std=WEIGHTS_STD)
            bert.pooler.dense.bias.data.zero_()
        else:
            # ----------
            # reintialize the top few blocks of the encoder
            # ----------
            bert_layer = bert.encoder.layer[-i_layer]
            reinitialize_bert_layer_recursively(bert_layer, WEIGHTS_STD)
    return


def reinitialize_bert_layer_recursively(bert_layer, weights_std):
    """Reinitialize the weights of a BERT layer recursively.
    Args:
        bert_layer (BertLayer): the BERT layer
        weights_std (float): the standard deviation of the normal distribution for initializing the weights of the layer layer
    """
    # base case: reaches the atomic layer
    if num_children_bert_layer(bert_layer) == 0:
        if isinstance(bert_layer, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            bert_layer.weight.data.normal_(mean=0.0, std=weights_std)
            if bert_layer.bias is not None:
                bert_layer.bias.data.zero_()
        # initialize the layernorm layer of bert
        # - https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        elif isinstance(bert_layer, nn.LayerNorm):
            bert_layer.bias.data.zero_()
            bert_layer.weight.data.fill_(1.0)
    # recursive case
    else:
        for layer in bert_layer.children():
            reinitialize_bert_layer_recursively(layer, weights_std)


def num_children_bert_layer(bert_layer):
    """Count the number of children of a BERT layer.
    Args:
        bert_layer (BertLayer): the BERT layer
    Returns:
        num_children (int): the number of children
    """
    num_children = 0
    for layer in bert_layer.children():
        num_children += 1
    return num_children


def specify_dict_dataset_eval(config, dict_dataset, partition_type):
    """Specify the datasets for evaluation while training.
    Args:
        config (dict): the config
        dict_dataset (dict): the dataset
        partition_type (str): the type of partition. Must be one of the following:
            - single_domain
            - gda
            - pseudo-labels
    Returns:
        dict_dataset_eval (dict): the dataset for evaluation
    """
    assert partition_type in ["single_domain", "gda", "pseudo-labels"]

    dict_dataset_eval = dict()
    if partition_type == "single_domain":
        if config.eval_on_train_raw == True:
            dict_dataset_eval["trainset_raw"] = dict_dataset["train_raw"]
        if config.eval_on_train_upsampled == True:
            dict_dataset_eval["trainset_upsampled"] = dict_dataset["train_upsampled"]
        if config.eval_on_train_downsampled == True:
            dict_dataset_eval["trainset_downsampled"] = dict_dataset["train_downsampled"]
        if config.eval_on_vali_raw == True:
            dict_dataset_eval["valiset_raw"] = dict_dataset["vali_raw"]
        if config.eval_on_vali_upsampled == True:
            dict_dataset_eval["valiset_upsampled"] = dict_dataset["vali_upsampled"]
        if config.eval_on_vali_downsampled == True:
            dict_dataset_eval["valiset_downsampled"] = dict_dataset["vali_downsampled"]

    elif partition_type == "gda":
        if config.eval_on_train_raw == True:
            dict_dataset_eval["trainset_raw"] = dict_dataset["source_train_raw"]
        if config.eval_on_train_upsampled == True:
            dict_dataset_eval["trainset_upsampled"] = dict_dataset["source_train_upsampled"]
        if config.eval_on_train_downsampled == True:
            dict_dataset_eval["trainset_downsampled"] = dict_dataset["source_train_downsampled"]
        if config.eval_on_vali_raw == True:
            dict_dataset_eval["valiset_raw"] = dict_dataset["source_vali_raw"]
        if config.eval_on_vali_upsampled == True:
            dict_dataset_eval["valiset_upsampled"] = dict_dataset["source_vali_upsampled"]
        if config.eval_on_vali_downsampled == True:
            dict_dataset_eval["valiset_downsampled"] = dict_dataset["source_vali_downsampled"]

    elif partition_type == "pseudo-labels":
        if config.eval_on_train_raw == True:
            dict_dataset_eval["trainset_raw"] = dict_dataset["train_raw"]
        if config.eval_on_train_upsampled == True:
            dict_dataset_eval["trainset_upsampled"] = dict_dataset["train_upsampled"]
        if config.eval_on_train_downsampled == True:
            dict_dataset_eval["trainset_downsampled"] = dict_dataset["train_downsampled"]
        if config.eval_on_vali_raw == True:
            dict_dataset_eval["valiset_raw"] = dict_dataset["vali_raw"]
        if config.eval_on_vali_upsampled == True:
            dict_dataset_eval["valiset_upsampled"] = dict_dataset["vali_upsampled"]
        if config.eval_on_vali_downsampled == True:
            dict_dataset_eval["valiset_downsampled"] = dict_dataset["vali_downsampled"]

    return dict_dataset_eval

def compute_metrics_sem_eval(eval_pred):
    metrics = compute_metrics_helper(eval_pred, ["f1", "accuracy", "recall", "precision"],
                                     by_class=True,
                                     num_classes=par_sem_eval.DICT_NUM_CLASS["SEM_EVAL"])
    return metrics

def func_compute_metrics_sem_eval():
    return compute_metrics_sem_eval


def compute_metrics_helper(eval_pred, list_metric, by_class=False, num_classes=None):
    """Compute metrics.

    Args:
        eval_pred (_type_): _description_
        list_metric (str): e.g., ["f1,accuracy,recall,precision","roc_auc"]
        by_class (bool, optional): Whether to compute metrics for each class (even if there are only two classes). Defaults to False.
        num_classes (int, optional): The number of classes. Only used when `by_class=True`. Defaults to None.
    Returns:
        metrics: _description_
    """
    for name_metric in list_metric:
        if name_metric not in ["f1", "accuracy", "recall", "precision", "roc_auc"]:
            raise ValueError("The metric name is not valid. Please choose from 'f1,accuracy,recall,precision','roc_auc'.")
    metric_computer = dict()
    for name_metric in list_metric:
        metric_computer[name_metric] = load_metric(name_metric)

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = dict()

    # compute metrics
    for name_metric in list_metric:

        if name_metric == "accuracy":
            metrics[name_metric] = metric_computer[name_metric].compute(predictions=predictions, references=labels)[name_metric]

        elif name_metric in ["f1", "recall", "precision"]:
            metrics[name_metric + "_macro"] = metric_computer[name_metric].compute(predictions=predictions, references=labels, average="macro")[name_metric]
            # compute by-class metrics
            if by_class:
                metrics[name_metric + "_byclass"] = metric_computer[name_metric].compute(predictions=predictions, references=labels,
                                                                                         labels=[i_class for i_class in range(num_classes)], average=None)[name_metric]
            # unpack the by-class scores
            for i_class in range(num_classes):
                metrics[name_metric + "_class_" + str(i_class)] = metrics[name_metric + "_byclass"][i_class]
            # remove the array-format by-class scores
            del metrics[name_metric + '_byclass']
    return metrics


def compute_metrics_without_trainer(labels, predictions, list_metric, dataset, logits=None, by_class=False, num_classes=None):
    """Compute the metrics for the predictions. Note that this can't be used to initiate a Trainer class because it only accepts `eval_pred` as an argument. Therefore, the argument "dataset" is not allowed. This function is only used for the standalone evaluation without Trainer, e.g., `gpt_evaluate_labels.py`.

    Args:
        labels (np.ndarray): the ground-truth labels
        predictions (np.ndarray): the predicted labels
        list_metric (str): e.g., ["f1,accuracy,recall,precision","roc_auc"]
        dataset (str): the dataset name, e.g., "COVID_VACCINE_Q1"
        logits (np.ndarray): the logits of the predictions. Only needed for the ROC-AUC score. Defaults to None.
        by_class (bool, optional): Whether to compute metrics for each class (even if there are only two classes). Defaults to False.
        num_classes (int, optional): The number of classes. Only used when `by_class=True`. Defaults to None.
    Returns:
        metrics (dict): the metrics
    """
    for name_metric in list_metric:
        if name_metric not in ["f1", "accuracy", "recall", "precision", "roc_auc"]:
            raise ValueError("The metric name is not valid. Please choose from 'f1,accuracy,recall,precision','roc_auc'.")
    metric_computer = dict()
    for name_metric in list_metric:
        metric_computer[name_metric] = load_metric(name_metric)

    metrics = dict()
    # compute metrics
    for name_metric in list_metric:
        if name_metric == "accuracy":
            metrics[name_metric] = metric_computer[name_metric].compute(predictions=predictions, references=labels)[name_metric]

        elif name_metric in ["f1", "recall", "precision"]:
            metrics[name_metric + "_macro"] = metric_computer[name_metric].compute(predictions=predictions, references=labels, average="macro")[name_metric]
            # compute by-class metrics
            if by_class:
                metrics[name_metric + "_byclass"] = metric_computer[name_metric].compute(predictions=predictions, references=labels,
                                                                                         labels=[i_class for i_class in range(num_classes)], average=None)[name_metric]
            # unpack the by-class scores
            for i_class in range(num_classes):
                metrics[name_metric + "_" + list(par.DICT_STANCES_CODE[dataset].keys())[i_class]] = metrics[name_metric + "_byclass"][i_class]

            # remove the array-format by-class scores
            del metrics[name_metric + '_byclass']
    return metrics


def upsample_df(df, col_name_label='label', factor_upsample=1, random_seed=0):
    """Upsample the non-majoirty classes such that all classes have the same number of samples.

    - match the size of each class to (factor_upsample * majority class size)
    -- 0 if no upsampling
    -- 1 if the majoruty class it not upsampled, and the other minority classes are upsampled to the same size as the majority class
    -- 2 if the majority class is upsampled twice, and the other minority classes are upsampled to twice the same size of the majority class

    Args:
        df (DataFrame): the dataset
        col_name_label (str): the column name of the class label. Default: 'label'
        factor_upsample (int): the factor to upsample the dataset. Default: 1
        random_seed (int): the random seed. Default: 0
    Returns:
        df_upsampled (DataFrame): the upsampled dataset
    """
    # count the number of each class
    list_count_classes = df[col_name_label].value_counts()
    # - find the "majority" class (so that other minority classes will be upsampled to this size)
    # -- the index
    ind_majority_class = list_count_classes.idxmax()
    # -- the size
    size_majority_class = list_count_classes[ind_majority_class]

    # the target size to upsample to
    # - all classes including the majority class will be upsampled to this size as well
    size_target = size_majority_class * factor_upsample

    # - create a new dataframe 'df_upsampled' where the minorities class are upsampled
    df_upsampled = pd.DataFrame()
    for ind_class in list_count_classes.index:
        df_class = df[df[col_name_label] == ind_class]
        # - upsample the minority class
        if ind_class != ind_majority_class:
            df_minority_class_upsampled = \
                resample(df_class, random_state=random_seed, n_samples=size_target, replace=True)
            df_upsampled = pd.concat([df_upsampled, df_minority_class_upsampled])
        else:
            # - repeat the majority class FACTOR_UPSAMPLE times
            df_majority_class_upsampled = pd.concat([df_class] * factor_upsample, ignore_index=True)
            df_upsampled = pd.concat([df_upsampled, df_majority_class_upsampled])

    # sanity check
    # - check if the class size is as expected
    list_count_classes_upsampled = df_upsampled[col_name_label].value_counts()
    for ind_class in list_count_classes.index:
        assert list_count_classes_upsampled[ind_class] == size_target

    return df_upsampled


def downsample_df(df, col_name_label='label', random_seed=0):
    """Downsample the non-minority classes to match the minority class such that all classes have the same number of samples.

    Args:
        df (DataFrame): the dataset
        col_name_label (str): the column name of the class label. Default: 'label'
        random_seed (int): the random seed. Default: 0
    Returns:
        df_downsampled (DataFrame): the downsampled dataset
    """
    # count the number of each class
    list_count_classes = df[col_name_label].value_counts()
    # - find the "minorty" class (so that other non-minority classes will be downsampled to this size)
    # -- the index
    ind_minority_class = list_count_classes.idxmin()
    # -- the size
    size_minority_class = list_count_classes[ind_minority_class]

    # the target size to downsampled to
    # - all classes will be downsampled to this size
    size_target = size_minority_class

    # - create a new dataframe 'df_downsampled' where the non-minority class are downsampled
    df_downsampled = pd.DataFrame()
    for ind_class in list_count_classes.index:
        df_class = df[df[col_name_label] == ind_class]
        # - downsample the minority class
        if ind_class != ind_minority_class:
            # - sampling without replacement to ensure that each tweet is unique
            df_minority_class_downsampled = \
                resample(df_class, random_state=random_seed, n_samples=size_target, replace=False)
            df_downsampled = pd.concat([df_downsampled, df_minority_class_downsampled])
        else:
            df_downsampled = pd.concat([df_downsampled, df_class])
    # sanity check
    # - check if all classes have the same number of samples
    assert df_downsampled[col_name_label].value_counts().nunique() == 1
    return df_downsampled


def evaluate_trained_trainer_over_sets(trainer, dataset, dict_dataset, col_name_set="set", return_predicted_labels=False,
                                       return_pred_label_type="df", keep_ts=False, keep_tweet_id=False, col_name_tweet_id="tweet_id"):
    """Evaluate a trained trainer over a list of sets. Also log to wandb etc. by calling `trainer.log(metrics)`.

    Args:
        trainer (Trainer): the trained trainer.
        dataset (str): the dataset, e.g., "COVID_VACCINE_Q1", "COVID_VACCINE_Q2".
        dict_dataset (dict): the dictionary of datasets to evaluate on.
        col_name_set (str): the column name of the set. Default: "set". Can be set to "time_domain" for gda.
        return_predicted_labels (bool): whether to add the predicted labels to dict_dataset. Default: False.
        return_pred_label_type (str): the type of the returned predicted labels. Default: "df". Can be set to "dataset" to return a dataset. Only used if `return_predicted_labels` is True.
        keep_ts (bool): whether to keep the timestamp. Default: False. Used only if `return_pred_label_type=="df"`.
        keep_tweet_id (bool): whether to keep the tweet id. Default: False. Used only if `return_pred_label_type=="df"`.
        col_name_tweet_id (str): the column name of the tweet id. Default: "tweet_id". Used only if `return_pred_label_type=="df"` and `keep_tweet_id==True`.
    Returns:
        df_metrics (DataFrame): the metrics over the sets.
        df_confusion_matrix (DataFrame): the confusion matrix over the sets.
        dict_dataset_or_df_with_pred (dict): the dictionary of datasets/dfs with predicted labels (as a new column in a dataset) if `return_predicted_labels` is True. Will return a dictionary of datasets if `return_pred_label_type` is "dataset", otherwise will return a dictionary of dfs.
            - https://discuss.huggingface.co/t/how-to-add-a-new-column-to-a-dataset/6453
    """
    assert dataset in ["COVID_VACCINE_Q1", "COVID_VACCINE_Q2", "COVID_VACCINE_Q2_v1", "COVID_VACCINE_Q2_v2", "WTWT", "SEM_EVAL"]
    # TODO
    assert return_pred_label_type in ["df", "dataset"]
    # ----------
    # Evaluate on each data set
    # ----------
    dict_metrics = dict()
    dict_num_all = dict()

    if "COVID_VACCINE" in dataset or dataset == "SEM_EVAL":
        dict_num_pos = dict()
        dict_num_neg = dict()
        dict_num_pos_pred = dict()
        dict_num_neg_pred = dict()
        if dataset == "COVID_VACCINE_Q2" or dataset == "SEM_EVAL":
            dict_num_neutral = dict()
            dict_num_neutral_pred = dict()

    elif dataset == "WTWT":
        dict_num_unrelated = dict()
        dict_num_support = dict()
        dict_num_refute = dict()
        dict_num_comment = dict()

        dict_num_unrelated_pred = dict()
        dict_num_support_pred = dict()
        dict_num_refute_pred = dict()
        dict_num_comment_pred = dict()

    dict_confusion_matrix = dict()

    if return_predicted_labels:
        dict_dataset_or_df_with_pred = dict()

    if return_pred_label_type == "df":
        # select the columns to keep while converting the dataset to a df
        # - "text", "month", "label", "predicted_label"
        list_col_to_keep = ["text", "label", "predicted_label"]
        if keep_ts:
            list_col_to_keep.append("timestamp")
        if keep_tweet_id:
            list_col_to_keep.append(col_name_tweet_id)

    for data_set in dict_dataset:
        # TODO: allow to evaluate across multiple domains (e.g., across the entire target domains)
        # - both micro and macro average
        # label_ids: the predicted probabilities
        # label_ids: the true labels
        predictions, label_ids, metrics = trainer.predict(dict_dataset[data_set], metric_key_prefix="test_" + data_set)
        # collect the true label distribution and the predicted label distribution
        label_pred = np.argmax(predictions, axis=-1)

        if return_predicted_labels:
            result_with_pred = dict_dataset[data_set].add_column("predicted_label", label_pred)
            if return_pred_label_type == "dataset":
                dict_dataset_or_df_with_pred[data_set] = result_with_pred
            elif return_pred_label_type == "df":
                # convert the dataset to a df
                dict_dataset_or_df_with_pred[data_set] = result_with_pred.select_columns(list_col_to_keep).to_pandas()
                # extract the month from 'data_set'
                # - e.g., "2020-01" from "target_2020-01" with regex
                month_found = re.search(r"(\d{4}-\d{2})", data_set)
                if month_found:
                    dict_dataset_or_df_with_pred[data_set]['month'] = month_found.group(1)
                else:
                    dict_dataset_or_df_with_pred[data_set]['month'] = data_set

        # log the testing metrics to wandb etc.
        trainer.log(metrics)
        # collect the metrics and write to csv
        dict_metrics[data_set] = metrics
        dict_num_all[data_set] = len(label_ids)

        if dataset == "COVID_VACCINE_Q1":
            dict_num_pos[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['YES'])
            dict_num_neg[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['NO'])
            dict_num_pos_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['YES'])
            dict_num_neg_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['NO'])
            dict_confusion_matrix[data_set] = confusion_matrix(
                label_ids, label_pred,
                # should be in sorted order from 0 to n_classes-1
                labels=[par.DICT_STANCES_CODE[dataset]['NO'],
                        par.DICT_STANCES_CODE[dataset]['YES']])
        elif dataset == "COVID_VACCINE_Q2" or dataset == "SEM_EVAL":
            dict_num_neutral[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['NONE'])
            dict_num_pos[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['FAVOR'])
            dict_num_neg[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['AGAINST'])
            dict_num_neutral_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['NONE'])
            dict_num_pos_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['FAVOR'])
            dict_num_neg_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['AGAINST'])
            dict_confusion_matrix[data_set] = \
                confusion_matrix(label_ids, label_pred,
                                 # should be in sorted order from 0 to n_classes-1
                                 labels=[par.DICT_STANCES_CODE[dataset]['NONE'],
                                         par.DICT_STANCES_CODE[dataset]['FAVOR'],
                                         par.DICT_STANCES_CODE[dataset]['AGAINST']])
        elif dataset == "COVID_VACCINE_Q2_v1":
            dict_num_pos[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['FAVOR'])
            dict_num_neg[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['NON-FAVOR'])
            dict_num_pos_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['FAVOR'])
            dict_num_neg_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['NON-FAVOR'])
            dict_confusion_matrix[data_set] = confusion_matrix(
                label_ids, label_pred,
                # should be in sorted order from 0 to n_classes-1
                labels=[par.DICT_STANCES_CODE[dataset]['NON-FAVOR'],
                        par.DICT_STANCES_CODE[dataset]['FAVOR']])
        elif dataset == "COVID_VACCINE_Q2_v2":
            dict_num_pos[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['AGAINST'])
            dict_num_neg[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['NON-AGAINST'])
            dict_num_pos_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['AGAINST'])
            dict_num_neg_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['NON-AGAINST'])
            dict_confusion_matrix[data_set] = confusion_matrix(
                label_ids, label_pred,
                # should be in sorted order from 0 to n_classes-1
                labels=[par.DICT_STANCES_CODE[dataset]['NON-AGAINST'],
                        par.DICT_STANCES_CODE[dataset]['AGAINST']])
        elif dataset == "WTWT":
            dict_num_unrelated[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['UNRELATED'])
            dict_num_support[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['SUPPORT'])
            dict_num_refute[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['REFUTE'])
            dict_num_comment[data_set] = sum(label_ids == par.DICT_STANCES_CODE[dataset]['COMMENT'])

            dict_num_unrelated_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['UNRELATED'])
            dict_num_support_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['SUPPORT'])
            dict_num_refute_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['REFUTE'])
            dict_num_comment_pred[data_set] = sum(label_pred == par.DICT_STANCES_CODE[dataset]['COMMENT'])

            dict_confusion_matrix[data_set] = \
                confusion_matrix(label_ids, label_pred,
                                 # should be in sorted order from 0 to n_classes-1
                                 labels=[par.DICT_STANCES_CODE[dataset]['UNRELATED'],
                                         par.DICT_STANCES_CODE[dataset]['SUPPORT'],
                                         par.DICT_STANCES_CODE[dataset]['REFUTE'],
                                         par.DICT_STANCES_CODE[dataset]['COMMENT']])

    # ----------
    # convert the metrics to a dataframe
    # ----------
    df_metrics = pd.DataFrame({
        col_name_set: [data_set for data_set in dict_metrics],
        "accuracy": [dict_metrics[data_set]["test_" + data_set + "_accuracy"] for data_set in dict_metrics]
        # "roc_auc": [dict_metrics[data_set][data_set + "_roc_auc"]["roc_auc"] for data_set in dict_metrics],
    })
    if dataset in ["COVID_VACCINE_Q1"]:
        # add the scores
        df_metrics["f1"] = \
            [dict_metrics[data_set]["test_" + data_set + "_f1"] for data_set in dict_metrics]
        df_metrics["recall"] = \
            [dict_metrics[data_set]["test_" + data_set + "_recall"] for data_set in dict_metrics]
        df_metrics["precision"] = \
            [dict_metrics[data_set]["test_" + data_set + "_precision"] for data_set in dict_metrics]
    elif dataset in ["COVID_VACCINE_Q2", "COVID_VACCINE_Q2_v1", "COVID_VACCINE_Q2_v2", "WTWT", "SEM_EVAL"]:
        # add the macro scores
        df_metrics["f1_macro"] = \
            [dict_metrics[data_set]["test_" + data_set + "_f1_macro"] for data_set in dict_metrics]
        df_metrics["recall_macro"] = \
            [dict_metrics[data_set]["test_" + data_set + "_recall_macro"] for data_set in dict_metrics]
        df_metrics["precision_macro"] = \
            [dict_metrics[data_set]["test_" + data_set + "_precision_macro"] for data_set in dict_metrics]
        # add the by-class scores
        for name_class, i_class in par.DICT_STANCES_CODE[dataset].items():
            df_metrics["f1_" + name_class] = \
                [dict_metrics[data_set]["test_" + data_set + "_f1_class_" + str(i_class)] for data_set in dict_metrics]
            df_metrics["precision_" + name_class] = \
                [dict_metrics[data_set]["test_" + data_set + "_precision_class_" + str(i_class)] for data_set in dict_metrics]
            df_metrics["recall_" + name_class] = \
                [dict_metrics[data_set]["test_" + data_set + "_recall_class_" + str(i_class)] for data_set in dict_metrics]

    # add the distribution of the labels
    df_metrics["num_all"] = [dict_num_all[data_set] for data_set in dict_num_all]
    if "COVID_VACCINE_Q2" in dataset or dataset == "SEM_EVAL":
        df_metrics["num_pos"] = [dict_num_pos[data_set] for data_set in dict_num_pos]
        df_metrics["num_neg"] = [dict_num_neg[data_set] for data_set in dict_num_neg]
        if dataset in ["COVID_VACCINE_Q1", "COVID_VACCINE_Q2_v1", "COVID_VACCINE_Q2_v2"]:
            df_metrics["num_pos_pred"] = [dict_num_pos_pred[data_set] for data_set in dict_num_pos_pred]
            df_metrics["num_neg_pred"] = [dict_num_neg_pred[data_set] for data_set in dict_num_neg_pred]
        elif dataset == "COVID_VACCINE_Q2" or dataset == "SEM_EVAL":
            df_metrics["num_neutral"] = [dict_num_neutral[data_set] for data_set in dict_num_neutral]
            df_metrics["num_pos_pred"] = [dict_num_pos_pred[data_set] for data_set in dict_num_pos_pred]
            df_metrics["num_neg_pred"] = [dict_num_neg_pred[data_set] for data_set in dict_num_neg_pred]
            df_metrics["num_neutral_pred"] = [dict_num_neutral_pred[data_set] for data_set in dict_num_neutral_pred]
    elif dataset == "WTWT":
        df_metrics["num_unrelated"] = [dict_num_support[data_set] for data_set in dict_num_unrelated]
        df_metrics["num_support"] = [dict_num_support[data_set] for data_set in dict_num_support]
        df_metrics["num_refute"] = [dict_num_refute[data_set] for data_set in dict_num_refute]
        df_metrics["num_comment"] = [dict_num_comment[data_set] for data_set in dict_num_comment]

        df_metrics["num_unrelated_pred"] = [dict_num_support_pred[data_set] for data_set in dict_num_unrelated_pred]
        df_metrics["num_support_pred"] = [dict_num_support_pred[data_set] for data_set in dict_num_support_pred]
        df_metrics["num_refute_pred"] = [dict_num_refute_pred[data_set] for data_set in dict_num_refute_pred]
        df_metrics["num_comment_pred"] = [dict_num_comment_pred[data_set] for data_set in dict_num_comment_pred]

    df_metrics = df_metrics.reset_index()
    # df_metrics = df_metrics.loc[['train', 'vali', 'test']]
    # round to 4 decimals
    df_metrics = df_metrics.round(4)
    # print(df_metrics)

    # ----------
    # convert the confusion matrices into a dataframe
    # ----------
    list_df_confusion = []
    for data_set in dict_confusion_matrix:
        df_confusion_matrix_this_set = pd.DataFrame(dict_confusion_matrix[data_set])
        df_confusion_matrix_this_set.columns = ["pred_" + key for key in par.DICT_STANCES_CODE[dataset].keys()]
        # prepend a column to indicate the true label
        df_confusion_matrix_this_set.insert(0, "true_class", ["true_" + key for key in par.DICT_STANCES_CODE[dataset].keys()])
        # prepend a column to indicate the data set
        df_confusion_matrix_this_set.insert(0, "data_set", data_set)
        list_df_confusion.append(df_confusion_matrix_this_set)
    df_confusion_matrix = pd.concat(list_df_confusion)

    if return_predicted_labels:
        return df_metrics, df_confusion_matrix, dict_dataset_or_df_with_pred
    else:
        return df_metrics, df_confusion_matrix


def extract_best_model_metrics(trained_trainer):
    """Extract the metrics of the best model from a trained trainer.

    Args:
        trained_trainer (Trainer): the trained trainer
    """

    # ----------
    # save the metrics of the best model
    # ----------
    metrics_best_model = dict()
    # get the metric used for selecting the best model
    # - e.g., "best_valiset_raw_f1_macro"=0.5
    metrics_best_model["best_criteria_{}".format(trained_trainer.args.metric_for_best_model)] = trained_trainer.state.best_metric
    # get the step of the best model
    best_model_step = int(re.search(r"checkpoint-(\d+)", trained_trainer.state.best_model_checkpoint).group(1))
    metrics_best_model["best_step"] = best_model_step
    # get the steps of each epoch
    steps_per_epoch = trained_trainer.state.max_steps / trained_trainer.args.num_train_epochs
    # get the epoch of the best model
    metrics_best_model["best_epoch"] = best_model_step / steps_per_epoch

    # get other metrics of the best model
    best_model_metrics = trained_trainer._get_metrics_from_log_history(best_model_step)
    for metric_name, metric_value in best_model_metrics.items():
        metrics_best_model["best_{}".format(metric_name)] = metric_value

    return metrics_best_model


def creat_dir_for_a_file_if_not_exists(file):
    """Create the directory of the file if it does not exist.

    Args:
        file (str): the file path
    """
    dir = path.dirname(file)
    if not path.exists(dir):
        makedirs(dir)


def get_dir_of_a_file(file):
    """Get the directory of the file.

    Args:
        file (str): the file path

    Returns:
        str: the directory of the file
    """
    return path.dirname(file)


def flatten_1d_nested_list(nested_list):
    """Flatten a 1D nested list.

    Args:
        nested_list (list): the nested list

    Returns:
        list: the flattened list
    - https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    """
    # return [item for sublist in nested_list for item in sublist]
    return list(chain.from_iterable(nested_list))


def remove_saved_models_in_checkpoint(path_run):
    """Remove the saved models in the checkpoint directory.

    Args:
        path_run (str): the path of the run directory. This is the parent directory of the checkpoint directories.
    """
    # delete the files at the checkpoint (except the json config files)
    # - find the files ending with .bin, .pt, .pth
    list_filetype = ['*.bin', '*.pt', '*.pth']
    list_files = []
    for file_type in list_filetype:
        list_files += glob(join(path_run, "checkpoint*", file_type))
    # - delete the files
    for file in list_files:
        os.remove(file)


def remove_checkpoint_dir(path_run):
    """Remove the checkpoint directory.

    Args:
        path_run (str): the path of the run directory. This is the parent directory of the checkpoint directories.
    """
    # delete the checkpoint dirs
    list_dirs = glob(join(path_run, "checkpoint*"))
    for dir in list_dirs:
        rmtree(dir)


def clean_empty_efevnts_dir(path_run):
    """Clean the empty efevents directories.

    Args:
        path_run (str): the path of the run directory. This is the parent directory of the efevents directories.
    """
    # - they are nested under a subdir (e.g., "1669923145.0353281")
    list_tfevent_files = glob(join(path_run, "runs", "**", "**", "events.out.tfevents*"))
    for tfevent_file in list_tfevent_files:
        os.remove(tfevent_file)


def save_dict_as_yaml(file_output_yaml, obj_dict):
    """Save a dictionary as a yaml file.

    Args:
        file_output_yaml (str): the output yaml file
        obj_dict (dict): the dictionary
    """
    with open(file_output_yaml, "w") as stream:
        yaml.dump(obj_dict, stream)


def compute_metric_from_confusion_matrix(np_confusion_matrix, metric_name, average_type=None, ind_positive_class=None):
    """Compute a metric from a confusion matrix.

    Args:
        np_confusion_matrix (np.array): the confusion matrix. The rows are the true labels, and the columns are the predicted labels.
        metric_name (str): the metric name
        average_type (str): the average type. Should be either "micro", "macro", "by_class". This will not be used when metric_name="accuracy".
        ind_positive_class (int|None): the index of the positive class. Only used when average_type="by_class".
    """

    assert metric_name in ["accuracy", "precision", "recall", "f1"]
    if metric_name != "accuracy":
        assert average_type in ["micro", "macro", "by_class"]
    num_class = np_confusion_matrix.shape[0]

    if metric_name == "accuracy":
        metric = np.trace(np_confusion_matrix) / np.sum(np_confusion_matrix)
    # macro: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    elif average_type == "macro":
        # - compute the metric for each class
        list_metrics = []
        for ind_class in range(num_class):
            metric = compute_metric_from_confusion_matrix(np_confusion_matrix, metric_name, average_type="by_class", ind_positive_class=ind_class)
            list_metrics.append(metric)
        # - compute the mean
        metric = np.mean(list_metrics)
    # micro: calculate metrics globally by counting the total true positives, false negatives and false positives.
    elif average_type == "micro":
        # micro-average: calculate metrics globally by counting the total true positives, false negatives and false positives.
        sum_tp, sum_fp, sum_tn, sum_fn = 0, 0, 0, 0
        for ind_class in range(num_class):
            tp, fp, tn, fn = \
                compute_tp_fp_tn_tn_from_confusion_matrix(np_confusion_matrix, ind_positive_class=ind_class)
            sum_tp += tp
            sum_fp += fp
            sum_tn += tn
            sum_fn += fn
        # - compute the metric
        if metric_name == "precision":
            metric = sum_tp / (sum_tp + sum_fp)
        elif metric_name == "recall":
            metric = sum_tp / (sum_tp + sum_fn)
        elif metric_name == "f1":
            precision = sum_tp / (sum_tp + sum_fp)
            recall = sum_tp / (sum_tp + sum_fn)
            metric = 2 * precision * recall / (precision + recall)
    # by_class: calculate metrics for a specific class.
    elif average_type == "by_class":
        tp, fp, tn, fn = \
            compute_tp_fp_tn_tn_from_confusion_matrix(np_confusion_matrix, ind_positive_class=ind_positive_class)
        if metric_name == "precision":
            metric = tp / (tp + fp)
        elif metric_name == "recall":
            metric = tp / (tp + fn)
        elif metric_name == "f1":
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            metric = 2 * precision * recall / (precision + recall)
    return metric


def compute_tp_fp_tn_tn_from_confusion_matrix(np_confusion_matrix, ind_positive_class):
    """Compute the true positives, false positives, true negatives and false negatives from a confusion matrix.

    Args:
        np_confusion_matrix (np.array): the confusion matrix. The rows are the true labels, and the columns are the predicted labels.
        ind_positive_class (int): the index of the positive class.
    Returns:
        tp (int): the true positives
        fp (int): the false positives
        tn (int): the true negatives
        fn (int): the false negatives
    """
    # - compute the true positives, false negatives and false positives
    tp = np_confusion_matrix[ind_positive_class, ind_positive_class]
    fn = np.sum(np_confusion_matrix[ind_positive_class, :]) - tp
    fp = np.sum(np_confusion_matrix[:, ind_positive_class]) - tp
    tn = np.sum(np_confusion_matrix) - tp - fn - fp
    return tp, fp, tn, fn


def seed_all(seed):
    """Seed all the random number generators.

    Args:
        seed (int): the seed
    """
    # random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    # th.cuda.manual_seed_all(seed)


def remove_oldest_instances_from_dataset(dataset, size_to_keep, col_name_timestamp="timestamp"):
    """Remove the oldest instances from a dataset based on the timestamp of the instances.

    Args:
        dataset (Dataset): the dataset. It should have a column named col_name_timestamp.
        size_to_keep (int): the number of instances to keep.
        col_name_timestamp (str): the name of the column containing the timestamp. Default: "timestamp".
    """
    if len(dataset) <= size_to_keep:
        return dataset

    # - get the cutoff timestamp
    list_timestamps = dataset[col_name_timestamp]
    # -- sort the timestamps
    # --- ['2021-05-17 12:00:46+00:00', ... ]
    list_timestamps_sorted = sorted([parse_timestamp_from_string(ts) for ts in list_timestamps])
    # list_timestamps = sorted(list_timestamps)
    # -- get the cutoff timestamp
    cutoff_timestamp = list_timestamps_sorted[-size_to_keep]

    # - filter the dataset by the cutoff timestamp
    dataset_reduced = dataset.filter(lambda x: parse_timestamp_from_string(x[col_name_timestamp]) > cutoff_timestamp)
    # note that due to that multiple tweets may share the same timestamp, the number of instances in the reduced dataset may be smaller than size_to_keep
    assert len(dataset_reduced) < size_to_keep, "len(dataset_reduced) = {}, size_to_keep = {}".format(len(dataset_reduced), size_to_keep)

    return dataset_reduced


def parse_timestamp_from_string(timestamp_str):
    """Parse a timestamp from a string.

    Args:
        timestamp_str (str): the timestamp in the format "2021-05-17 12:00:46+00:00", or, if failed,  "2021-05-17".
    Returns:
        timestamp (datetime.datetime): the timestamp.
    """
    # - parse the timestamp
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S%z")
    except ValueError:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")
    return timestamp


def sort_list_by_occurrence(list_values):
    """Sort a list by the occurrence of the values.

    Args:
        list_values (list): the list of values.
    Returns:
        list_values_sorted (list): the list of values sorted by the occurrence of the values.
    - see: https://stackoverflow.com/questions/23429426/sorting-a-list-by-frequency-of-occurrence-in-a-list
    """
    return [item for items, c in Counter(list_values).most_common() for item in [items] * c]


def convert_time_unit_into_name(time_unit):
    """Convert a time unit into a name.

    Args:
        time_unit (str|tuple(str)): the time unit, e.g., '2020-12' or ('2020-12', '2021-01').
    Returns:
        name (str): the name.
    """
    if isinstance(time_unit, str):
        name = time_unit
    elif isinstance(time_unit, tuple):
        if len(time_unit) == 1:
            name = time_unit[0]
        else:
            name = time_unit[0] + "_" + time_unit[-1]
    else:
        raise ValueError("time_unit = {}".format(time_unit))
    return name


def parse_list_to_list_or_str(list_values):
    """Parse a list to a list or a string.

    Args:
        list_values (list): the list.
    Returns:
        list_values_or_str (list|str): the list or the string.
    """
    if isinstance(list_values, list):
        if len(list_values) > 1:
            list_values = list_values
        else:
            # - if the list contains only one element, return the element
            list_values = list_values[0]
    else:
        list_values = list_values
    return list_values


def wait_for_user_input():
    while True:
        user_input = input("Do you want to proceed? (y/n)")
        if user_input.lower() == "y":
            return True
        elif user_input.lower() == "n":
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def glob_re(pattern, strings):
    """
    Glob with regular expressions.
    - https://stackoverflow.com/a/51246151/8060278
    """
    return filter(re.compile(pattern).match, strings)


def list_full_paths(directory):
    """https://www.askpython.com/python/examples/python-directory-listing"""
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def generate_list_of_months(start_month, end_month):
    """Generate a list of months.

    Args:
        start_month (str): the start month in the format "YYYY-MM".
        end_month (str): the end month in the format "YYYY-MM".
    Returns:
        list_of_months (list): the list of months.
    """
    date_range = pd.date_range(start=start_month, end=end_month, freq='M')

    # Convert the DatetimeIndex to a list of strings with the format 'YYYY-MM'
    list_of_months = date_range.strftime('%Y-%m').tolist()


def check_if_item_exist_in_nested_list(item, nested_list):
    """Check if an item exists in a nested list.

    Args:
        item (any): the item.
        nested_list (list): the nested list.
    Returns:
        bool: True if the item exists in the nested list, False otherwise.
    """
    for sublist in nested_list:
        if item in sublist:
            return True
    return False


def logits_to_log_probs(logits):
    return th.log_softmax(logits, dim=-1)


def add_version_suffix_to_file_name(file_name, version_name, version_suffix):
    return file_name.replace(version_name, version_name + version_suffix)


def index_logits_by_1d_index_tensor(outputs_logits, index_tensor_1d):
    """ Index the logits tensor by another tensor of an 1d index. This is equivalent to retriveing the logits of each token over a sequence (or, "path"). Can be computed by forward() and generate() functions of the flan-t5 model.

    Args:
        outputs_logits (torch.Tensor): the tensor of logits to index. Shape: (batch_size, num_tokens_in_dictionary, len_output_tokens).
        index_tensor_1d (torch.Tensor): the 1d index tensor. Shape: (batch_size x len_output_tokens).
    Returns:
        log_probs (torch.Tensor): the tensor of log-probabilities. Shape: (batch_size, len_output_tokens).
        sum_log_probs (torch.Tensor): the tensor of sum of log-probabilities. Shape: (batch_size).
    """
    assert isinstance(outputs_logits, th.Tensor)
    assert isinstance(index_tensor_1d, th.Tensor)
    assert outputs_logits.shape[0] == index_tensor_1d.shape[0]
    assert outputs_logits.shape[2] == index_tensor_1d.shape[1]

    device = outputs_logits.device
    batch_size, num_tokens_in_dictionary, len_output_tokens = outputs_logits.shape
    index_tensor_1d = index_tensor_1d.to(device)

    log_probs = logits_to_log_probs(outputs_logits)
    # - index the log-probabilities by `outputs_ids`
    # -- (batch_size)
    # -- prepare indices for the first and third dimensions
    index_dim_batch = th.arange(batch_size).view(-1, 1).expand(batch_size, len_output_tokens).to(device)
    index_dim_output_token = th.arange(len_output_tokens).view(1, -1).expand(batch_size, len_output_tokens).to(device)
    combined_indices = th.stack([index_dim_batch, index_tensor_1d, index_dim_output_token], dim=-1)
    # - (batch_size x len_output_tokens)
    log_probs_path = th.gather(log_probs, 1, combined_indices[:, :, 1].unsqueeze(1)).squeeze()
    # - sum over the tokens
    sum_log_probs = log_probs_path.sum(dim=-1)
    # assert log_probs_path.shape == batch_size
    return log_probs_path, sum_log_probs


def convert_stance_code_to_text(code_stance, dataset):
    """ Convert the code of stance to the text of stance.
    """
    return par.DICT_STANCES_CODE_INV[dataset][code_stance]


def tidy_name(raw_name):
    """ Tidy the name of a file or a directory by replacing all non-alphanumeric characters with underscores.
    """
    tidy_name = re.sub(r'\W', '_', raw_name)
    return tidy_name

def display_resized_image_in_notebook(file_image, scale=1):
    """ Display an image in a notebook.
    """
    # - https://stackoverflow.com/questions/69654877/how-to-set-image-size-to-display-in-ipython-display
    # Open the image
    image = PILImage.open(file_image)
    display(image.resize((int(image.width * scale), int(image.height * scale))))

class AttrDict(dict):
    # https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
