"""
The parameters specific to the SemEval dataset.
@author: Yun-Shiuan Chuang
@contact: yunshiuan.chuang@gmail.com
"""
from os.path import join
import pandas as pd
from parameters_meta import ParametersMeta as par_meta


class ParametersSemEval2016():
    # par
    # -------------
    # SemEval 2016
    # -------------
    DATASET_META = "SEM_EVAL"
    TEXT_ID = "ID"
    # - version for the (raw) input data
    VERSION_DATA_RAW = '20230415'
    # - version for the (processed) input data
    # VERSION_DATA = '20230330'
    VERSION_DATA = '20230415'

    # - version for the NLP model result
    DICT_VERSION_OUTPUT = {
        # -- single_domain_baseline
        "single_domain_baseline": "v1",

        # -- gpt
        "gpt": {
            # --- gpt-chat-turbo-3_5
            "chatgpt_turbo_3_5": "v3",

            # --- gpt_3_davinci
            "gpt_3_davinci": "v1",

            # --- flan-t5-xxl
            "flan-t5-xxl": "v2",

            # --- flan_ul2
            "flan_ul2": "v1"
        }
    }

    # -- gpt
    # --- the version of the prompt
    VERSION_PROMPT = "v2"

    # --- gpt-chat-turbo-3_5
    MODEL_GPT = "chatgpt_turbo_3_5"
# 
    # --- gpt_3_davinci
    # MODEL_GPT = "gpt_3_davinci"
    #
    # --- gpt_3_davinci
    # MODEL_GPT = "gpt_3_davinci"

    # --- flan-t5-xxl
    # MODEL_GPT = "flan-t5-xxl"
    # --- flan_ul2
    # MODEL_GPT = "flan_ul2"

    # - convert topic name
    DICT_TOPICS_RAW_TO_NEW = {
        "SEM_EVAL": {
            "Atheism": "Atheism",
            "Climate Change is a Real Concern": "Climate",
            "Feminist Movement": "Feminist",
            "Hillary Clinton": "Clinton",
            "Legalization of Abortion": "Abortion"
        }
    }
    DICT_TOPICS_NEW_TO_RAW = {
        "SEM_EVAL": {
            "Atheism": "Atheism",
            "Climate": "Climate Change is a Real Concern",
            "Feminist": "Feminist Movement",
            "Clinton": "Hillary Clinton",
            "Abortion": "Legalization of Abortion"
        }
    }
    # - convert topic name to number
    DICT_TOPICS_CODE = {
        "SEM_EVAL": {
            "Atheism": 0,
            "Climate": 1,
            "Feminist": 2,
            "Clinton": 3,
            "Abortion": 4
        }
    }
    DICT_TOPICS_COLOR = {
        "SEM_EVAL": {
            "Atheism": "red",
            "Climate Change is a Real Concern": "blue",
            "Feminist Movement": "green",
            "Hillary Clinton": "orange",
            "Legalization of Abortion": "purple"
        }
    }
    # - the number of topics
    DICT_NUM_TOPICS = {
        "SEM_EVAL": len(DICT_TOPICS_CODE["SEM_EVAL"])
    }

    # - note: the code should always start from 0
    DICT_STANCES_RAW_TO_NEW = {
        "SEM_EVAL": {
            'NONE': 'NONE',
            'FAVOR': 'FAVOR',
            'AGAINST': 'AGAINST'
        }
    }
    # - convert stance label to number
    DICT_STANCES_CODE = {
        "SEM_EVAL": {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2}
    }
    # convert number to stance label
    DICT_STANCES_CODE_INV = {
        "SEM_EVAL": {0: 'NONE', 1: 'FAVOR', 2: 'AGAINST'}
    }
    DICT_NUM_CLASS = {
        "SEM_EVAL": len(DICT_STANCES_CODE["SEM_EVAL"])
    }
    DICT_STANCES_COLOR = {
        "SEM_EVAL": {
            'NONE': 'grey',
            'FAVOR': 'blue',
            'AGAINST': 'red'
        }
    }
    
    # - recode the stance label (e.g., collapsing several classes into one)
    # -- applied to the processed data
    DICT_STANCES_LABEL_RECODE_PROCSSED = {
        # "SEM_EVAL": {
        #     'FAVOR': 'FAVOR',
        #     'NONE': 'NON-FAVOR',
        #     'AGAINST': 'NON-FAVOR'}
    }
    # -- applied to the raw data
    DICT_STANCES_LABEL_RECODE_RAW = {
        # "SEM_EVAL": {
        #     '1-POSITIVE': '1-POSITIVE',
        #     '0-NEUTRAL or UNCLEAR': '0-NEUTRAL or UNCLEAR',
        #     '2-NEGATIVE': '0-NEUTRAL or UNCLEAR'}
    }
    # -- applied to the gpt labels (3-class stances)
    DICT_STANCES_LABEL_RECODE_GPT = {
        "gpt_3_davinci": {
            # "SEM_EVAL": {
            #     'In favor': 'FAVOR',
            #     'Neutral': 'NON-FAVOR',
            #     'Against': 'NON-FAVOR'}
        },
        "flan-t5-xxl": {
            # "SEM_EVAL": {
            #     'in-favor': 'FAVOR',
            #     'neutral': 'NON-FAVOR',
            #     'against': 'NON-FAVOR'}
        },
        "flan_ul2": {
            # "SEM_EVAL": {
            #     'in-favor': 'FAVOR',
            #     'neutral': 'NON-FAVOR',
            #     'against': 'NON-FAVOR'
            # }
        }
    }

    # - code usd by GPT
    DICT_STANCES_CODE_GPT = {
        "gpt_3_davinci": {
            # - produced by GPT
            # "SEM_EVAL": {'Neutral': 0, 'In favor': 1, 'Against': 2},
        },
        "chatgpt_turbo_3_5": {
            "SEM_EVAL": {'neutral': 0, 'in-favor': 1, 'against': 2,
                     "neutral-or-unclear": 0, 'unclear': 0}
        },
        "flan-t5-xxl": {
            "SEM_EVAL": {'neutral': 0, 'in-favor': 1, 'against': 2,
                                 "neutral-or-unclear": 0},
        },
        "flan-t5-large": {
            "SEM_EVAL": {'neutral': 0, 'in-favor': 1, 'against': 2,
                                 "neutral-or-unclear": 0},
        },
        "flan-t5-small": {
            "SEM_EVAL": {'neutral': 0, 'in-favor': 1, 'against': 2,
                                 "neutral-or-unclear": 0},
        },                
        "flan_ul2": {
            # "SEM_EVAL": {'neutral': 0, 'in-favor': 1, 'against': 2,
            #                      "neutral-or-unclear": 0}
        }
    }

    # - ratio used for partitioning the dataset into train, dev, and test
    # -- in this dataset, the test set is given in the raw data, so we only need to partition the train and dev set
    DICT_RATIO_PARTITION_TRAIN_VALI = {"train": 0.8, "vali": 0.2}

    # path
    PATH_ROOT = par_meta.PATH_ROOT
    PATH_DATA = par_meta.PATH_DATA
    PATH_RESULT = par_meta.PATH_RESULT

    # - SemEval
    PATH_DATA_SEM_EVAL = join(PATH_DATA, "semeval_2016")
    PATH_DATA_SEM_EVAL_RAW = join(PATH_DATA_SEM_EVAL, "raw")
    PATH_RESULT_SEM_EVAL = join(PATH_RESULT, "semeval_2016")
    PATH_RESULT_SEM_EVAL_TUNING = join(PATH_RESULT_SEM_EVAL, "tuning")
    PATH_RESULT_SEM_EVAL_LLM = join(PATH_RESULT_SEM_EVAL, "llm")

    PATH_ERROR_ANALYSIS_SEM_EVAL_REPORT = join(PATH_RESULT_SEM_EVAL, "error_analysis")
    PATH_ERROR_ANALYSIS_SEM_EVAL_REPORT_THIS_VERSION = join(PATH_ERROR_ANALYSIS_SEM_EVAL_REPORT, VERSION_DATA)

    # -- results across different model types
    PATH_RESULT_SEM_EVAL_ACROSS_MODEL_TYPES = join(PATH_RESULT_SEM_EVAL, "across_model_types")
    PATH_RESULT_SEM_EVAL_ACROSS_MODEL_TYPES_SINGLE_DOMAIN = join(PATH_RESULT_SEM_EVAL_ACROSS_MODEL_TYPES, "single_domain")

    # -- single_domain_baseline
    PATH_RESULT_SEM_EVAL_SINGLE_DOMAIN_BASELINE = join(PATH_RESULT_SEM_EVAL, "single_domain_baseline")
    PATH_RESULT_SEM_EVAL_SINGLE_DOMAIN_BASELINE_THIS_VERSION = \
        join(PATH_RESULT_SEM_EVAL_SINGLE_DOMAIN_BASELINE,
             DICT_VERSION_OUTPUT["single_domain_baseline"])
    # -- gpt
    PATH_RESULT_SEM_EVAL_GPT = join(PATH_RESULT_SEM_EVAL, "gpt", MODEL_GPT)
    PATH_RESULT_SEM_EVAL_GPT_THIS_VERSION = join(PATH_RESULT_SEM_EVAL_GPT, DICT_VERSION_OUTPUT["gpt"][MODEL_GPT])

    # file
    # - input
    # -- raw
    FILE_DATA_SEM_EVAL_RAW_TRAIN = join(
        PATH_DATA_SEM_EVAL_RAW,
        "task_a_train_" + VERSION_DATA_RAW + ".txt")
    FILE_DATA_SEM_EVAL_RAW_TEST = join(
        PATH_DATA_SEM_EVAL_RAW,
        "task_a_test_" + VERSION_DATA_RAW + ".txt")
    # -- processed
    FILE_DATA_SEM_EVAL_PROCESSED = join(
        PATH_DATA_SEM_EVAL,
        "processed",
        VERSION_DATA + ".csv")

    DICT_FILE_DATA_SEM_EVAL_PROCESSED = {
        "SEM_EVAL": FILE_DATA_SEM_EVAL_PROCESSED
    }
    # -- processed (partitions)
    # --- the partition information for train/dev/test
    FILE_DATA_SEM_EVAL_PROCESSED_PARTITIONS = join(
        PATH_DATA_SEM_EVAL,
        "processed",
        VERSION_DATA + "_partitions.csv")

    DICT_FILE_DATA_SEM_EVAL_PROCESSED_PARTITIONS = {
        "SEM_EVAL": FILE_DATA_SEM_EVAL_PROCESSED_PARTITIONS
    }
    # -- embedded for GPT
    FILE_DATA_SEM_EVAL_PROCESSED_EMBEDDED = join(
        PATH_DATA_SEM_EVAL,
        "gpt_processed",
        VERSION_DATA + "_embedded_" + VERSION_PROMPT + ".csv")
    DICT_FILE_DATA_SEM_EVAL_PROCESSED_EMBEDDED = {
        "SEM_EVAL": FILE_DATA_SEM_EVAL_PROCESSED_EMBEDDED
    }

    # -- labels generated by GPT
    FILE_DATA_SEM_EVAL_PROCESSED_GPT_LABELS = join(
        PATH_RESULT_SEM_EVAL_GPT_THIS_VERSION,
        VERSION_DATA + "_labeled_" + DICT_VERSION_OUTPUT["gpt"][MODEL_GPT] + ".csv")
    DICT_FILE_DATA_SEM_EVAL_PROCESSED_GPT_LABELS = {
        "SEM_EVAL": FILE_DATA_SEM_EVAL_PROCESSED_GPT_LABELS
    }
    # - output
    # -- model training result (metrics)
    # -- single_domain_baseline
    FILE_RESULT_SEM_EVAL_MODEL_SINGLE_DOMAIN_BASELINE_METRIC = join(
        PATH_RESULT_SEM_EVAL_SINGLE_DOMAIN_BASELINE_THIS_VERSION, "metric_over_sets.csv"
    )
    DICT_RESULT_SEM_EVAL_MODEL_SINGLE_DOMAIN_BASELINE_METRIC = {
        "SEM_EVAL": FILE_RESULT_SEM_EVAL_MODEL_SINGLE_DOMAIN_BASELINE_METRIC
    }

    # -- gpt
    # --- single_domain
    FILE_RESULT_SEM_EVAL_MODEL_GPT_METRIC_SINGLE_DOMAIN = join(
        PATH_RESULT_SEM_EVAL_GPT_THIS_VERSION, "metric_over_sets.csv"
    )
    DICT_RESULT_SEM_EVAL_MODEL_GPT_METRIC_SINGLE_DOMAIN = {
        "SEM_EVAL": FILE_RESULT_SEM_EVAL_MODEL_GPT_METRIC_SINGLE_DOMAIN
    }
    # -- model training result (confusion matrix)
    # --- single_domain_baseline
    FILE_RESULT_SEM_EVAL_MODEL_SINGLE_DOMAIN_BASELINE_CONFUSION_MATRIX = join(
        PATH_RESULT_SEM_EVAL_SINGLE_DOMAIN_BASELINE_THIS_VERSION, "confusion_matrix_over_sets.csv"
    )
    DICT_RESULT_SEM_EVAL_MODEL_SINGLE_DOMAIN_BASELINE_CONFUSION_MATRIX = {
        "SEM_EVAL": FILE_RESULT_SEM_EVAL_MODEL_SINGLE_DOMAIN_BASELINE_CONFUSION_MATRIX
    }
    # --- gpt
    # ---- single_domain
    FILE_RESULT_SEM_EVAL_MODEL_GPT_CONFUSION_MATRIX_SINGLE_DOMAIN = join(
        PATH_RESULT_SEM_EVAL_GPT_THIS_VERSION, "confusion_matrix_over_sets.csv"
    )
    DICT_RESULT_SEM_EVAL_MODEL_GPT_CONFUSION_MATRIX_SINGLE_DOMAIN = {
        "SEM_EVAL": FILE_RESULT_SEM_EVAL_MODEL_GPT_CONFUSION_MATRIX_SINGLE_DOMAIN
    }

    # -- data describer result
    # --- label distribution (csv)
    FILE_CSV_DATA_SEM_EVAL_LABEL_DISTRIBUTION = join(
        PATH_DATA_SEM_EVAL,
        "data_description", VERSION_DATA + "_label_distribution.csv")
    DICT_FILE_CSV_DATA_SEM_EVAL_LABEL_DISTRIBUTION = {
        "SEM_EVAL": FILE_CSV_DATA_SEM_EVAL_LABEL_DISTRIBUTION
    }

    # --- label distribution (png)
    FILE_PLOT_DATA_SEM_EVAL_LABEL_DISTRIBUTION = join(
        PATH_DATA_SEM_EVAL,
        "data_description", VERSION_DATA + "_label_distribution.png")
    DICT_FILE_PLOT_DATA_SEM_EVAL_LABEL_DISTRIBUTION = {
        "SEM_EVAL": FILE_PLOT_DATA_SEM_EVAL_LABEL_DISTRIBUTION
    }
    # --- topic distribution (csv)
    FILE_CSV_DATA_SEM_EVAL_TOPIC_DISTRIBUTION = join(
        PATH_DATA_SEM_EVAL,
        "data_description", VERSION_DATA + "_topic_distribution.csv")
    DICT_FILE_CSV_DATA_SEM_EVAL_TOPIC_DISTRIBUTION = {
        "SEM_EVAL": FILE_CSV_DATA_SEM_EVAL_TOPIC_DISTRIBUTION
    }
    # --- topic distribution (png)
    FILE_PLOT_DATA_SEM_EVAL_TOPIC_DISTRIBUTION = join(
        PATH_DATA_SEM_EVAL,
        "data_description", VERSION_DATA + "_topic_distribution.png")
    DICT_FILE_PLOT_DATA_SEM_EVAL_TOPIC_DISTRIBUTION = {
        "SEM_EVAL": FILE_PLOT_DATA_SEM_EVAL_TOPIC_DISTRIBUTION
    }
    # -- error analysis report
    FILE_ERROR_ANALYSIS_SEM_EVAL = join(
        PATH_ERROR_ANALYSIS_SEM_EVAL_REPORT_THIS_VERSION, "error_analysis_report.csv"
    )
    DICT_FILE_ERROR_ANALYSIS_SEM_EVAL = {
        "SEM_EVAL": FILE_ERROR_ANALYSIS_SEM_EVAL
    }
