"""
Data loader and embed texts in prompts for the GPT.
@author: Yun-Shiuan Chuang
@contact: yunshiuan.chuang@gmail.com
"""
import pandas as pd
import numpy as np
import re
from data_processor import SemEvalDataProcessor
from utils import get_parameters_for_dataset, creat_dir_for_a_file_if_not_exists


class GPTDataProcessor():
    def embed_prompt(self, path_input, path_output, write_csv=True, return_df=False):
        """Embed the processed data in prompts.

        Args:
            file_input (str): the processed data
            file_output (str): the embedded processed data
            write_csv (bool, optional): whether to write the embedded data to a csv file. Defaults to True.
            return_df (bool, optional): whether to return the embedded data as a dataframe. Defaults to False.
        """
        raise NotImplementedError()

    def recode_classes(self, file_input, file_output, file_type):
        """Recode the classes of GPT predictions. E.g., collapsing several classes into one.

        Args:
            file_input (str): the file containing the labels or the GPT predictions before recoding
            file_output (str): the file containing the labels or the GPT predictions after recoding
            file_type (str): "GPT_predictions", "embedded_tweets"
        """
        raise NotImplementedError()

    def read_preprocessed_tweets(self, file_preprocessed_tweets):
        # read the preprocessed data (unembedded)
        raise NotImplementedError()

    def read_embedded_tweets(self, file_embedded_tweets):
        # read the raw data
        raise NotImplementedError()

    def read_gpt_labels(self, file_gpt_labels):
        # read the preprocessed data
        raise NotImplementedError()


class SemEvalGPTDataProcessor(GPTDataProcessor):
    """Embed the processed data of the SemEval tweets in prompts.
    - see data_processor.py for the data processing.
    Args:
        DataProcessor (_type_): _description_
    """

    def __init__(self, version_prompt, prompt=None, topic="all") -> None:
        """_summary_

        Args:
            topic (str): the topic of interest, e.g., "Abortion".
            version_prompt (str): the version of the prompt, e.g., "v1".
            prompt (str, optional): The prompt used to embed the tweets. Defaults to None. When not specified, the default prompt will be used.
            topic (str, optional): the topic of interest, e.g., "Abortion". Defaults to "all".
        """
        self.par = get_parameters_for_dataset("SEM_EVAL")
        self.dataset = "SEM_EVAL"
        assert topic in self.par.DICT_TOPICS_RAW_TO_NEW[self.dataset].values(
        ) or topic == "all", "The topic should be one of the topics in the dataset or 'all'."
        # get the default prompt based on the version of the prompt
        if version_prompt == "zero_shot":
            DICT_PROMPT = dict()
            for topic_raw, topic_new in self.par.DICT_TOPICS_RAW_TO_NEW[self.dataset].items():
                DICT_PROMPT[topic_new] = "What is the stance of the tweet below with respect to '{}'?  If we can infer from the tweet that the tweeter supports '{}', please label it as 'in-favor'. If we can infer from the tweet that the tweeter is against '{}', please label is as 'against'. If we can infer from the tweet that the tweeter has a neutral stance towards '{}', please label it as 'neutral-or-unclear'. If there is no clue in the tweet to reveal the stance of the tweeter towards '{}', please also label is as 'neutral-or-unclear'. Please use exactly one word from the following 3 categories to label it: 'in-favor', 'against', 'neutral-or-unclear'.".format(
                    topic_raw, topic_raw, topic_raw, topic_raw, topic_raw)
                DICT_PROMPT[topic_new] += " Here is the tweet. '{}' The stance of the tweet is: "
            self.prompt = DICT_PROMPT
            self.num_examples = 0
            self.list_tweet_id_in_prompt = []
        elif version_prompt == "few_shot":
            for topic_raw, topic_new in self.par.DICT_TOPICS_RAW_TO_NEW[self.dataset].items():
                DICT_PROMPT = dict()
                if topic_new != "Abortion":
                    # yet to be implemented for other topics
                    pass
                DICT_PROMPT[topic_new] = "What is the stance of the tweet below with respect to '{}'?  If we can infer from the tweet that the tweeter supports '{}', please label it as 'in-favor'. If we can infer from the tweet that the tweeter is against '{}', please label is as 'against'. If we can infer from the tweet that the tweeter has a neutral stance towards '{}', please label it as 'neutral-or-unclear'. If there is no clue in the tweet to reveal the stance of the tweeter towards '{}', please also label is as 'neutral-or-unclear'. Please use exactly one word from the following 3 categories to label it: 'in-favor', 'against', 'neutral-or-unclear'.".format(
                    topic_raw, topic_raw, topic_raw, topic_raw, topic_raw)
                DICT_PROMPT[topic_new] += "  Here are some examples of tweets. Make sure to classify the last tweet correctly.\n"
                DICT_PROMPT[topic_new] = \
                    DICT_PROMPT[topic_new] +\
                    "Q: Tweet: it's a free country. freedom includes freedom of choice.\n" +\
                    'Is this tweet in-favor, against, or neutral-or-unclear?\n' +\
                    'A: in-favor\n' +\
                    "Q: Tweet: i really don't understand how some people are pro-choice. a life is a life no matter if it's 2 weeks old or 20 years old.\n" +\
                    'Is this tweet in-favor, against, or neutral-or-unclear?\n' +\
                    'A: against\n' +\
                    "Q: Tweet: so ready for my abortion debate\n" +\
                    'Is this tweet in-favor, against, or neutral-or-unclear?\n' +\
                    'A: neutral-or-unclear\n' +\
                    "Q: Tweet: '{}'\n" +\
                    'Is this tweet in-favor, against, or neutral-or-unclear?\n' +\
                    'A: '
            self.num_examples = 3
            self.list_tweet_id_in_prompt = ["2368", "2312", "2381"]
        elif version_prompt == "CoT":
            DICT_PROMPT = dict()
            for topic_raw, topic_new in self.par.DICT_TOPICS_RAW_TO_NEW[self.dataset].items():
                DICT_PROMPT[topic_new] = "What is the stance of the tweet below with respect to '{}'?  If we can infer from the tweet that the tweeter supports '{}', please label it as 'in-favor'. If we can infer from the tweet that the tweeter is against '{}', please label is as 'against'. If we can infer from the tweet that the tweeter has a neutral stance towards '{}', please label it as 'neutral-or-unclear'. If there is no clue in the tweet to reveal the stance of the tweeter towards '{}', please also label is as 'neutral-or-unclear'.".format(
                    topic_raw, topic_raw, topic_raw, topic_raw, topic_raw)
                DICT_PROMPT[topic_new] += " Here is the tweet. '{}'"
                DICT_PROMPT[topic_new] += " What is the stance of the tweet with respect to '{}'? Please make sure that at the end of your response, use exactly one word from the following 3 categories to label the stance with respect to '{}': 'in-favor', 'against', 'neutral-or-unclear’. Let's think step by step.".format(
                    topic_raw, topic_raw)
            self.prompt = DICT_PROMPT
            self.num_examples = 0
            self.list_tweet_id_in_prompt = []
        else:
            raise ValueError("The version of the prompt is not supported.")
        if prompt is not None:
            self.prompt = prompt
        elif topic == "all":
            self.prompt = DICT_PROMPT
        elif topic in DICT_PROMPT.keys():
            self.prompt = DICT_PROMPT[topic]
        else:
            raise ValueError("The topic is not supported.")

        self.version_prompt = version_prompt
        data_processor = SemEvalDataProcessor()
        self.data_processor = data_processor
        self.file_processed_tweet = self.data_processor._get_file_processed_default(topic)
        self.topic = topic

    def embed_prompt(self, file_output=None, write_csv=True, return_df=False, write_sample_csv=False):
        """Embed the processed data in prompts.

        Args:
            file_output (str, optional): the output file. Defaults to None. Only required if write_csv is True.
            write_csv (bool, optional): whether to write the embedded data to a csv file. Defaults to True.
            return_df (bool, optional): whether to return the embedded data as a dataframe. Defaults to False.
            write_sample_csv (bool, optional): whether to write a sample of the embedded data to a csv file. Defaults to False.
        """
        if write_csv or write_sample_csv:
            assert file_output is not None, "The output file is required if write_csv or write_sample_csv is True."
        # read the processed data
        df_processed = self.data_processor._read_preprocessed_data(file_preprocessed_data=self.file_processed_tweet)
        # remove the first column
        if "Unnamed: 0" in df_processed.columns:
            df_processed = df_processed.drop(columns=["Unnamed: 0"])
        # embed the tweets in the prompt
        df_processed["tweet_embedded"] = df_processed["tweet"].apply(lambda x: self.prompt.format(x))

        # check if the examples indeed exist in the 'tweet' column
        # - true is the count is 2
        if self.num_examples > 0:
            assert df_processed.apply(
                lambda x: x["tweet_embedded"].count(x["tweet"]) == 2, axis=1).sum() == self.num_examples

        if write_csv or write_sample_csv:
            creat_dir_for_a_file_if_not_exists(file_output)
            if self.topic != "all":
                file_output = file_output.replace("embedded", "embedded_{}".format(self.topic))
        # save the embedded data
        if write_csv:
            df_processed.to_csv(file_output)
        if write_sample_csv:
            df_processed.head(n=30).to_csv(file_output.replace(".csv", "_sample.csv"))
        if return_df:
            return df_processed

    def read_preprocessed_tweets(self, file_preprocessed_tweets):
        # read the preprocessed data
        df_processed = pd.read_csv(file_preprocessed_tweets,
                                   dtype={self.par.TEXT_ID: str})
        self.df_processed = df_processed
        return df_processed

    def read_embedded_tweets(self, file_embedded_tweets):
        # read the embedded tweets
        df_embedded_tweets = pd.read_csv(file_embedded_tweets,
                                         dtype={self.par.TEXT_ID: str})
        self.df_embedded_tweets = df_embedded_tweets
        return df_embedded_tweets

    def read_gpt_labels(self, file_gpt_labels):
        # read the preprocessed data
        df_gpt_labels = pd.read_csv(file_gpt_labels,
                                    dtype={self.par.TEXT_ID: str})
        self.df_gpt_labels = df_gpt_labels
        return df_gpt_labels

    def _get_file_predictions_default(self, topic="all"):
        """Get the default file containing the GPT predictions.

        Args:
            topic (str, optional): the topic. Defaults to "all".
        Returns:
            str: the file containing the GPT predictions
        """
        file_predictions_default = self.par.DICT_FILE_DATA_SEM_EVAL_PROCESSED_GPT_LABELS[self.dataset].replace(
            ".csv", "_" + topic + ".csv")
        return file_predictions_default

    def _get_list_tweet_id_in_prompt(self):
        """Get the list of tweet ids in the prompt.

        Returns:
            list: the list of tweet ids in the prompt
        """
        return self.list_tweet_id_in_prompt

    def _get_num_examples_in_prompt(self):
        """Get the number of examples in the prompt.

        Returns:
            int: the number of examples in the prompt
        """
        return self.num_examples


if __name__ == "__main__":
    pass