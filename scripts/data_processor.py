"""
Data loader and processor.
@author: Yun-Shiuan Chuang
@contact: yunshiuan.chuang@gmail.com
"""
import pandas as pd
import numpy as np
import emoji
import re
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from os import listdir
from os.path import join
from utils import convert_time_unit_into_name, get_parameters_for_dataset, glob_re, list_full_paths, creat_dir_for_a_file_if_not_exists, check_if_item_exist_in_nested_list, get_dir_of_a_file


class DataProcessor():
    def preprocess(self, path_input, path_output):
        raise NotImplementedError()

    def recode_classes(self, file_input, file_output, file_input_type):
        raise NotImplementedError()


class SemEvalDataProcessor(DataProcessor):
    """Process the raw data of the SemEval-2016 tweets.
    - output columns
        - created_at
        - merger
        - stance
    - remove stop words
        - "RT @user_id"
        - url (http...)
        - see `VERSION_LOG_DATA` for details
    Args:
        DataProcessor (_type_): _description_
    """

    def __init__(self) -> None:
        """
        """
        self.par = get_parameters_for_dataset("SEM_EVAL")
        self.dataset = "SEM_EVAL"
        self.file_raw_train_default = self.par.FILE_DATA_SEM_EVAL_RAW_TRAIN
        self.file_raw_test_default = self.par.FILE_DATA_SEM_EVAL_RAW_TEST
        self.file_processed_default = self.par.FILE_DATA_SEM_EVAL_PROCESSED
        self.file_partitions_default = self.par.DICT_FILE_DATA_SEM_EVAL_PROCESSED_PARTITIONS[self.dataset]

    def preprocess(self, file_input=None, file_output=None):
        """Preprocess raw data into processed data using the majority rule.

        Args:
            file_input (str): The file path of the raw data. If None, use the default file path. Default: None.
            file_output (str): The file path of the processed data. If None, use the default file path. Default: None.
        """
        # read the raw data
        if file_input is None:
            file_input_train = self.file_raw_train_default
            file_input_test = self.file_raw_test_default
        df_raw = self._read_raw_data(file_input_train, file_input_test)
        # drop the unnamed column
        if "Unnamed: 0" in df_raw.columns:
            df_raw = df_raw.drop(columns=["Unnamed: 0"])

        # -----------------------
        # build the processed df
        # -----------------------
        df_processed = pd.DataFrame(
            {"ID": df_raw[self.par.TEXT_ID],
             "tweet": df_raw.Tweet,
             "topic": df_raw.Target,
             "label": df_raw.Stance,
             "partition": df_raw.partition}
        )

        # - mutate the stance variable and the topic variable
        # -- from raw to new categories using self.par.DICT_STANCES_RAW_TO_NEW["SEM_EVAL"]
        # -- useful when collapsing the stance categories etc.
        if self.dataset == "SEM_EVAL":
            df_processed['label'] =\
                df_processed['label'].apply(lambda x: self.par.DICT_STANCES_RAW_TO_NEW[self.dataset][x])
            assert sorted(df_processed["label"].unique()) == \
                sorted(list(self.par.DICT_STANCES_RAW_TO_NEW[self.dataset].values()))
            df_processed['topic'] =\
                df_processed['topic'].apply(lambda x: self.par.DICT_TOPICS_RAW_TO_NEW[self.dataset][x])
            assert sorted(df_processed["topic"].unique()) == \
                sorted(list(self.par.DICT_TOPICS_RAW_TO_NEW[self.dataset].values()))
        else:
            raise Exception("Invalid dataset: " + self.dataset)
        # -----------------------
        # process the text
        # -----------------------
        # to lower case
        df_processed["tweet_processed"] = df_processed.tweet.str.lower()

        # -----------------------
        # preprocessing rules
        # - rules of preprocessing
        #   - to retain important information
        #     - keep "mentions" if the uername is wuth high frequncy. Otherwise, replace the mention with the [USERNAME].
        #       - keep top 5% of the usernames
        #       - in previous version, I attempted to remove all the mentions (though some weren't successfully removed)
        #   - to reduce noise
        #     - use python' `emoji` library to parse emoticons and emojis
        #   - to reduce noise and count of tokens
        #     - keep only alphabets, numbers, punctuations, emojis (this remove non-English characters)
        #       - pattern = `r'[\w\s,!.?;:\-\(\)\[\]\'\"<>]+|[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+'`
        #         - The first part [\w\s,!.?;:\-\(\)\[\]\'\"<>]+ matches any combination of alphanumeric characters (\w), whitespace (\s), punctuation marks, and common HTML/XML characters.
        #         - The second part [\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+ matches any emojis from the Unicode code ranges of smileys and people, symbols and signs, transport and map symbols, and flags respectively.
        #           - `\U0001F600-\U0001F64F`: Emoticons (smileys, people, animals, nature, food, activities)
        #           - `\U0001F300-\U0001F5FF`: Miscellaneous symbols and pictographs (clocks, arrows, shapes, weather, buildings, transport, flags)
        #           - `\U0001F680-\U0001F6FF`: Transport and map symbols (cars, trains, boats, planes, signs)
        #           - `\U0001F1E0-\U0001F1FF`: Regional indicator symbols (flags of countries)
        #     - replace double (or more) quotes with single quotes
        #       - `"` -> `'`
        #       - `""` -> `'`
        #     - replace apostrophes with single quotes
        #       - `’` -> `'`
        #     - remove repetitions in space
        #     - remove empty tweets (after the above preprocessing)
        # -----------------------
        # remove "#semst" at the end of the tweet
        df_processed["tweet_processed"] = df_processed.tweet_processed.str.replace(pat='#semst$', repl='', regex=True)
        # remove RT
        df_processed["tweet_processed"] = df_processed.tweet_processed.str.replace(pat='(rt @)\\w+(\\s|:)', repl='', regex=True)
        # remove urls
        df_processed["tweet_processed"] = df_processed.tweet_processed.str.replace(pat='http\\S+|www.\\S+', repl='', regex=True)
        # parse emojis
        df_processed["tweet_processed"] = df_processed.tweet_processed.apply(lambda x: emoji.demojize(x))
        # remove leading and trailing spaces
        df_processed["tweet_processed"] = df_processed.tweet_processed.str.replace(pat='^\\s+|\\s+$', repl='', regex=True)
        # remove repetitions in space
        df_processed["tweet_processed"] = df_processed.tweet_processed.str.replace(pat='\\s+', repl=' ', regex=True)

        # replace double (or more) quotes with single quotes
        df_processed["tweet_processed"] = df_processed.tweet_processed.str.replace(pat='"', repl="'", regex=False)

        # replace apostrophes with single quotes
        df_processed["tweet_processed"] = df_processed.tweet_processed.str.replace(pat='’', repl="'", regex=False)
        # remove backward slash
        df_processed["tweet_processed"] = df_processed.tweet_processed.str.replace(pat='\\', repl='', regex=False)
        # keep only alphabets, numbers, punctuations, emojis (this remove non-English characters)
        df_processed["tweet_processed"] = df_processed.tweet_processed.str.replace(
            pat=r'[^a-zA-Z0-9\s,!.?;:@#\-\(\)\[\]\'\"<>]+|[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+', repl='', regex=True)

        # find and keep the top 3% of mentions
        list_mentions = df_processed.tweet_processed.str.findall(pat='@\\w+')
        # - flatten the list
        list_mentions = [item for sublist in list_mentions for item in sublist]
        # - count the frequency of each mention
        counter_mentions = Counter(list_mentions)
        # - convert the counter to a dataframe with column "mention" and "count"
        df_mentions = pd.DataFrame(counter_mentions.most_common(), columns=["mention", "count"])
        # - keep the top 3% of the mentions
        # -- get the count of the 3% of the mentions
        count_mentions = df_mentions["count"].quantile(q=0.97)
        # -- keep the mentions with count >= count_mentions
        list_mentions_keep = df_mentions[df_mentions["count"] >= count_mentions]["mention"].tolist()
        # - replace the mentions with 'USERNAME' if the mention is not in the list_mentions_keep
        df_processed["tweet_processed"] = df_processed.tweet_processed.apply(lambda x: re.sub(r'@\w+', '@USERNAME', x) if x not in list_mentions_keep else x)
        # -----------------------
        # remove tweets with duplicated text
        #   - may due to retweets etc.
        # -----------------------
        df_processed = df_processed.drop_duplicates(subset=["tweet_processed"], keep="first")

        # -----------------------
        # remove empty tweets
        # -----------------------
        df_processed = df_processed[df_processed["tweet_processed"] != ""]

        # -----------------------
        # rename the column
        # -----------------------
        df_processed["tweet"] = df_processed["tweet_processed"]
        df_processed = df_processed.drop(columns=["tweet_processed"])
        # save the processed data
        if file_output is None:
            file_output = self.file_processed_default

        creat_dir_for_a_file_if_not_exists(file_output)
        df_processed.to_csv(file_output)

        # save the list of mentions
        df_mentions_keep = df_mentions[df_mentions["mention"].isin(list_mentions_keep)]
        df_mentions_keep.to_csv(file_output.replace(".csv", "_mentions_kept.csv"), index=False)

    def recode_classes(self, file_input, file_output, file_input_type="processed"):
        """Recode the classes of a processed. E.g., collapsing several classes into one.

        Args:
            file_input (str): The file path of the input file before class-recoding.
            file_output (str): The file path of the output file with recoded classes.
            file_input_type (str, optional): The type of the input file. It can be either "raw" or "processed". Defaults to "processed".
        """
        assert file_input_type in ["raw", "processed"]

        if file_input_type == "processed":
            # read the processed data
            df_processed = self._read_preprocessed_data(file_input)

            # recode the classes based on the dictionary
            df_processed["label_majority_recoded"] =\
                df_processed["label_majority"].map(self.par.DICT_STANCES_LABEL_RECODE_PROCSSED[self.dataset])

            # check if there are any missing values
            assert df_processed["label_majority_recoded"].isna().sum() == 0

            df_processed = df_processed.drop(columns=["label_majority"])
            df_processed = df_processed.rename(columns={"label_majority_recoded": "label_majority"})
            df_processed = df_processed.drop(columns=["Unnamed: 0"])
            # save the recoded data
            df_processed.to_csv(file_output)

        elif file_input_type == "raw":
            # read the raw data
            df_raw = self._read_raw_data(file_input)

            # recode the classes based on the dictionary
            for col_name in self.par.NAMES_RATERS:
                df_raw[col_name] =\
                    df_raw[col_name].map(self.par.DICT_STANCES_LABEL_RECODE_RAW[self.dataset])

            # check if the unique values in the df is the same as the dictionary
            # - ignoring the nan values
            assert sorted(df_raw[self.par.NAMES_RATERS].unstack().unique().astype(str)) ==\
                sorted(list(set(self.par.DICT_STANCES_LABEL_RECODE_RAW[self.dataset].values()))) + ['nan']

            # save the recoded data
            df_raw.to_csv(file_output)

    def _read_preprocessed_data(self, file_preprocessed_data=None, topic="all"):
        """Read the preprocessed data.
        Args:
            file_preprocessed_data (str): The file path of the preprocessed data. If None, the file path will be self.file_processed_default.
            topic (str): The topic to read. If "all", all topics will be read. Defaults to "all".

        Returns:
            df_processed (pd.DataFrame): the preprocessed data.
        """
        if file_preprocessed_data is None:
            file_preprocessed_data = self.file_processed_default
        df_processed = pd.read_csv(file_preprocessed_data,
                                   dtype={self.par.TEXT_ID: str})
        if "Unnamed: 0" in df_processed.columns:
            df_processed = df_processed.drop(columns=["Unnamed: 0"])
        # filter the data by topic
        if topic != "all":
            df_processed = df_processed[df_processed["topic"] == topic]
        self.df_processed = df_processed
        return df_processed

    def _read_raw_data(self, file_raw_data_train=None, file_raw_data_test=None, read_train=True, read_test=True, topic="all"):
        """Read the raw data.
        Args:
            file_raw_data_train (str): The file path of the raw train data. If None, the file path will be self.file_raw_train_default. Defaults to None.
            file_raw_data_test (str): The file path of the raw test data. If None, the file path will be self.file_raw_test_default. Defaults to None.
            read_train (bool): Whether to read the train data. Defaults to True.
            read_test (bool): Whether to read the test data. Defaults to True.
            topic (str): The topic to read. If "all", all topics will be read. Defaults to "all".

        Returns:
            df_raw (pd.DataFrame): the raw data.
        """
        assert read_train or read_test, "Either read_train or read_test must be True."
        assert topic in ["all"] + list(self.par.DICT_TOPICS_CODE[self.dataset].keys()), "The topic must be in {}.".format(["all"] + list(self.par.DICT_TOPICS_CODE[self.dataset].keys()))
        if read_train:
            if file_raw_data_train is None:
                file_raw_data_train = self.file_raw_train_default
        if read_test:
            if file_raw_data_test is None:
                file_raw_data_test = self.file_raw_test_default
        # read the raw data while retaining the partition information
        # - https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas
        if read_train:
            df_raw_train = pd.read_csv(file_raw_data_train,
                                       delimiter="\t", encoding='unicode_escape',
                                       dtype={self.par.TEXT_ID: str})
            df_raw_train["partition"] = "train"
        if read_test:
            df_raw_test = pd.read_csv(file_raw_data_test,
                                      delimiter="\t", encoding='unicode_escape',
                                      dtype={self.par.TEXT_ID: str})
            df_raw_test["partition"] = "test"

        df_raw = pd.DataFrame()
        if read_train:
            df_raw = pd.concat([df_raw, df_raw_train])
        if read_test:
            df_raw = pd.concat([df_raw, df_raw_test])
        self.df_raw = df_raw
        assert df_raw[self.par.TEXT_ID].is_unique
        assert df_raw[self.par.TEXT_ID].notna().sum() == df_raw.shape[0]
        assert df_raw[self.par.TEXT_ID].notna().sum() == df_raw[self.par.TEXT_ID].drop_duplicates().shape[0]
        if read_train:
            assert (df_raw["partition"] == "train").sum() == df_raw_train.shape[0]
        if read_test:
            assert (df_raw["partition"] == "test").sum() == df_raw_test.shape[0]
        
        # filter by topic
        if topic != "all":
            # should map by DICT_TOPICS_NEW_TO_RAW
            df_raw = df_raw[df_raw["Target"] == self.par.DICT_TOPICS_NEW_TO_RAW[self.dataset][topic]]    
        return df_raw

    def partition_processed_data(self, file_preprocessed_data=None, seed=42, verbose=True):
        """
        Partition the processed data:
            - each topic is partitioned into train, dev, and test.
            - within each topic, the label distribution should be constant across train, dev, and test.
            - note that in this dataset, the test set is given in the raw data, so we only need to partition the train and dev set.

        Args:
            file_processed (str): The file path of the output file with information about the partition. If None, the file path will be self.file_processed_default. Defaults to None.
            seed (int, optional): The seed for the random number generator. Defaults to 42.
            verbose (bool, optional): Whether to print the partition information. Defaults to True.
        Returns:
            df_partitions (pd.DataFrame): The dataframe with Tweet IDs within each partition.
                columns: ["tweet_id", "label_majority", "partition", "time_unit"]
            seed (int): The seed for the random number generator.
        """
        # read the processed data
        df_processed = self._read_preprocessed_data(file_preprocessed_data)
        if file_preprocessed_data is None:
            file_preprocessed_data = self.file_processed_default
        # create a dataframe to store the partition information
        df_partitions = pd.DataFrame(columns=[self.par.TEXT_ID, "label", "partition", "topic"])

        # --------------------
        # Iterate through each topic
        # --------------------
        # record the maximum difference between the ratio of the any class between train/vali/test
        max_ratio_diff = 0
        for topic_this in self.par.DICT_TOPICS_CODE[self.dataset]:

            # filter in the data for this topic
            df_processed_this = df_processed[df_processed["topic"] == topic_this]
            # --------------------
            # Partition the data into train, dev
            # - while keeping the label distribution constant
            # --------------------
            # filter out the test set
            df_processed_this_train = df_processed_this[df_processed_this["partition"] == "train"]
            # Split the data into train and vali sets
            # - while keeping the label distribution constant
            X = df_processed_this_train[[self.par.TEXT_ID, "topic"]]
            y = df_processed_this_train["label"]
            train_vali_split = StratifiedShuffleSplit(n_splits=1,
                                                      test_size=self.par.DICT_RATIO_PARTITION_TRAIN_VALI["vali"], random_state=seed)
            for train_index, vali_test_index in train_vali_split.split(X, y):
                X_train, X_vali = X.iloc[train_index], X.iloc[vali_test_index]
                y_train, y_vali = y.iloc[train_index], y.iloc[vali_test_index]
            # --------------------
            # Select the test set
            # --------------------
            df_processed_this_test = df_processed_this[df_processed_this["partition"] == "test"]
            X_test = df_processed_this_test[[self.par.TEXT_ID, "topic"]]
            y_test = df_processed_this_test["label"]

            # sanity check
            assert len(X_train) + len(X_vali) + len(X_test) == len(df_processed_this)
            assert len(y_train) + len(y_vali) + len(y_test) == len(df_processed_this)
            # check the ratio of the label distribution
            if verbose:
                for y_class in self.par.DICT_STANCES_CODE[self.dataset]:
                    ratio_train = len(y_train[y_train == y_class]) / len(y_train)
                    ratio_vali = len(y_vali[y_vali == y_class]) / len(y_vali)
                    ratio_test = len(y_test[y_test == y_class]) / len(y_test)
                    # print(f"ratio_train: {ratio_train}")
                    # print(f"ratio_vali: {ratio_vali}")
                    # print(f"ratio_test: {ratio_test}")
                    # print(f"ratio_train - ratio_vali: {ratio_train - ratio_vali}")
                    # print(f"ratio_train - ratio_test: {ratio_train - ratio_test}")
                    # print(f"ratio_vali - ratio_test: {ratio_vali - ratio_test}")
                    ratio_diff_train_vali = abs(ratio_train - ratio_vali) / ratio_vali
                    ratio_diff_train_test = abs(ratio_train - ratio_test) / ratio_test
                    ratio_diff_vali_test = abs(ratio_vali - ratio_test) / ratio_test
                    # assert ratio_diff_train_vali < 0.1
                    # assert ratio_diff_train_test < 0.1
                    # assert ratio_diff_vali_test < 0.1
                    # print the max of this topic
                    max_ratio_diff_this_topic = max(ratio_diff_train_vali, ratio_diff_train_test, ratio_diff_vali_test)
                    print("Topic: {}, y_class:{}, ratio_diff_train_vali:{}".format(topic_this, y_class, ratio_diff_train_vali))
                    print("Topic: {}, y_class:{}, ratio_diff_train_test:{}".format(topic_this, y_class, ratio_diff_train_test))
                    print("Topic: {}, y_class:{}, ratio_diff_vali_test:{}".format(topic_this, y_class, ratio_diff_vali_test))
                    # print("Topic: {}, y_class:{}, max_ratio_diff_this_topic:{}".format(topic_this, y_class,max_ratio_diff_this_topic))
                    max_ratio_diff = max(max_ratio_diff, ratio_diff_train_vali, ratio_diff_train_test, ratio_diff_vali_test)

            # --------------------
            # create a dataframe to store the partitioned df
            # --------------------
            df_partitions_this_train = pd.DataFrame(
                {
                    self.par.TEXT_ID: X_train[self.par.TEXT_ID],
                    "topic": X_train["topic"],
                    "label": y_train,
                    "partition": "train"
                }
            )
            df_partitions_this_vali = pd.DataFrame(
                {
                    self.par.TEXT_ID: X_vali[self.par.TEXT_ID],
                    "topic": X_vali["topic"],
                    "label": y_vali,
                    "partition": "vali",
                }
            )
            df_partitions_this_test = pd.DataFrame(
                {
                    self.par.TEXT_ID: X_test[self.par.TEXT_ID],
                    "topic": X_test["topic"],
                    "label": y_test,
                    "partition": "test",
                }
            )
            df_partitions_this = pd.concat(
                [df_partitions_this_train, df_partitions_this_vali, df_partitions_this_test],
                axis=0)
            # collect the partitioned df
            df_partitions = pd.concat([df_partitions, df_partitions_this], axis=0)
        # print the max ratio difference
        if verbose:
            print(f"max_ratio_diff: {max_ratio_diff}")
        # check the partition ratio
        list_ratio_partition = [len(df_partitions[df_partitions["partition"] == "train"]) / len(df_partitions),
                                len(df_partitions[df_partitions["partition"] == "vali"]) / len(df_partitions),
                                len(df_partitions[df_partitions["partition"] == "test"]) / len(df_partitions)]
        for i, partition in enumerate(["train", "vali"]):
            assert round(100 * ((list_ratio_partition[i] / sum(list_ratio_partition[:2])) - self.par.DICT_RATIO_PARTITION_TRAIN_VALI[partition])) == 0

        # write the partitioned df to a file
        file_output = file_preprocessed_data.replace(".csv", "_partitions.csv")
        df_partitions.to_csv(file_output, index=False)
        return df_partitions

    def read_partitions(self, file_partitions=None, topic="all"):
        """Read the partitioned data from a file.

        Args:
            file_partitions (str, optional): Path to the file containing the partitioned data. If None, the default path will be used. Defaults to None.
            topic (str, optional): Topic to filter. Defaults to "all".

        Returns:
            pd.DataFrame: Partitioned data.
        """
        if file_partitions is None:
            file_partitions = self._get_file_partitions_default(topic=topic)
        df_partitions = pd.read_csv(file_partitions,
                                    dtype={self.par.TEXT_ID: str})
        self.df_partitions = df_partitions
        return df_partitions

    def filter_preprocessed_by_topic(self, topic, file_preprocessed_data=None, file_partitions=None):
        """Filter the preprocessed data and the partitioned mask by a topic.

        Args:
            topic (str): Topic to filter.
            file_preprocessed_data (str, optional): Path to the file containing the preprocessed data. If None, the default path will be used. Defaults to None.
            file_partitions (str, optional): Path to the file containing the partitioned data. If None, the default path will be used. Defaults to None.

        Returns:
            pd.DataFrame: Filtered data.
        """
        assert topic in self.par.DICT_TOPICS_CODE[self.dataset]
        if file_preprocessed_data is None:
            file_preprocessed_data = self.file_processed_default
        if file_partitions is None:
            file_partitions = self.file_partitions_default
        df_preprocessed = self._read_preprocessed_data(file_preprocessed_data)
        df_partitions = self.read_partitions(file_partitions)
        df_preprocessed_this = df_preprocessed[df_preprocessed["topic"] == topic]
        df_partitions_this = df_partitions[df_partitions["topic"] == topic]

        # write the filtered data to a file
        file_preprocessed_data_topic = file_preprocessed_data.replace(".csv", f"_{topic}.csv")
        df_preprocessed_this.to_csv(file_preprocessed_data_topic, index=False)
        file_partitions_topic = file_partitions.replace(".csv", f"_{topic}.csv")
        df_partitions_this.to_csv(file_partitions_topic, index=False)
        return df_preprocessed_this, df_partitions_this

    def _get_file_processed_default(self, topic="all"):
        assert topic in list(self.par.DICT_TOPICS_CODE[self.dataset].keys()) + ["all"]
        if topic == "all":
            file_processed_default = self.par.FILE_DATA_SEM_EVAL_PROCESSED
        else:
            file_processed_default = self.par.FILE_DATA_SEM_EVAL_PROCESSED
            file_processed_default = file_processed_default.replace(".csv", "_{}.csv".format(topic))
        return file_processed_default

    def _get_file_partitions_default(self, topic="all"):
        assert topic in list(self.par.DICT_TOPICS_CODE[self.dataset].keys()) + ["all"]
        if topic == "all":
            file_partitions_default = self.par.DICT_FILE_DATA_SEM_EVAL_PROCESSED_PARTITIONS[self.dataset]
        else:
            file_partitions_default = self.par.DICT_FILE_DATA_SEM_EVAL_PROCESSED_PARTITIONS[self.dataset]
            file_partitions_default = file_partitions_default.replace(".csv", "_{}.csv".format(topic))
        return file_partitions_default


if __name__ == "__main__":
    SEED = 42
    par = get_parameters_for_dataset()
    VERSION_DATA = par.VERSION_DATA
    DATASET_META = par.DATASET_META
    TOPIC_OF_INTEREST = "Abortion"
    # sem_eval_data = SemEvalDataProcessor()
    # df_raw_train = sem_eval_data._read_raw_data(read_train=True,read_test=False,topic=TOPIC_OF_INTEREST)
    sem_eval_data = SemEvalDataProcessor()
    sem_eval_data.preprocess()
    sem_eval_data.partition_processed_data(seed=SEED)
    sem_eval_data.filter_preprocessed_by_topic(topic=TOPIC_OF_INTEREST)

    print("----------")
    print("Complete!")
    print("----------")
