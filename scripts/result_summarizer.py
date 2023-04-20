"""
Summarize run results (e.g., metrics) after model training.
@author: Yun-Shiuan Chuang
@contact: yunshiuan.chuang@gmail.com
"""
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
from math import ceil
import re
from yaml import safe_load
import argparse

from utils import compute_metric_from_confusion_matrix, compute_tp_fp_tn_tn_from_confusion_matrix, convert_time_unit_into_name, parse_list_to_list_or_str, get_parameters_for_dataset


class ResultSummarizer():
    def __init__(self, dataset, list_version_output, task, model_type, eval_mode, path_input_root=None, path_output=None, use_post_processed_predictions=False,file_name_metrics=None,file_name_confusion_mat=None) -> None:
        """
        Args:
            dataset (str): "COVID_VACCINE_Q1" or "COVID_VACCINE_Q2"
            list_version_output (list(str)): e.g., ["v4","v5"]
            task (str): e.g., "Q1", "Q2"
            model_type (str): e.g., "gda_baseline", "single_domain_baseline"
            eval_mode (str): "gda" or "single_domain".
            path_input_root (str): path to the root of the input dir., e.g., "results/covid_vaccine/gda_baseline". It's will recursively search for the input dir for each version. If None, it will use the default path. Default: None.
            path_output (str): path to the output folder, e.g., "results/covid_vaccine/gda_baseline/summary/Q2/metrics_over_domains". If None, it will use the default path. Default: None.
            use_post_processed_predictions (bool): whether to use the post-processed predictions. Default: False.
            file_name_metrics (str): file name of the metrics. If None, it will use the default file name. Default: None.
            file_name_confusion_mat (str): file name of the confusion matrix. If None, it will use the default file name. Default: None.
        """
        assert model_type in ["gda_baseline", "gda_pseudo_cumulative", "gda_pseudo_buffer",
                              "gda_pseudo_all",
                              "single_domain_baseline"] or model_type.startswith("llm_")
        assert eval_mode in ["gda", "single_domain"]
        self.par = get_parameters_for_dataset()
        self.dataset = dataset
        self.list_version_output = list_version_output
        self.task = task
        self.eval_mode = eval_mode
        self.model_type = model_type
        if path_input_root is None:
            path_input_root = self._get_default_path_input_root()
        if path_output is None:
            path_output = self._get_default_path_output()
        self.path_input_root = path_input_root
        self.path_output = path_output
        self.use_post_processed_predictions = use_post_processed_predictions
        # ----------
        # Read in the metrics data over domains
        # ----------
        # read in all the metrics data over all versions of this task
        df_metrics_over_domains = pd.DataFrame()
        for version in self.list_version_output:
            input_path_this_version = self._get_input_path_this_version(version)
            if file_name_metrics:
                file_metrics = join(input_path_this_version, file_name_metrics)
            else:
                if self.eval_mode == "gda":
                    file_metrics = join(input_path_this_version, "metric_over_domains.csv")
                elif self.eval_mode == "single_domain":
                    file_metrics = join(input_path_this_version, "metric_over_sets.csv")
            # if use post-processed predictions, suffix the file name
            if self.use_post_processed_predictions:
                file_metrics = file_metrics.replace(".csv", "_post_processed.csv")
            df_metrics = pd.read_csv(file_metrics)

            # drop `Unnamed: 0` if exists
            if "Unnamed: 0" in df_metrics.columns:
                df_metrics = df_metrics.drop(columns=["Unnamed: 0"])
            # drop `index` if exists
            if "index" in df_metrics.columns:
                df_metrics = df_metrics.drop(columns=["index"])
            df_metrics["version"] = version
            df_metrics["task"] = task
            df_metrics_over_domains = pd.concat([df_metrics_over_domains, df_metrics], ignore_index=True)
        self.df_metrics_over_domains = df_metrics_over_domains
        # ----------
        # Read in the confusion matrix over domains
        # ----------
        # read in all the confusion matrix over all versions of this task
        df_confusion_mat_over_domains = pd.DataFrame()
        for version in self.list_version_output:
            input_path_this_version = self._get_input_path_this_version(version)
            if file_name_confusion_mat:
                file_confusion_mat = join(input_path_this_version, file_name_confusion_mat)
            else:
                if self.eval_mode == "gda":
                    file_confusion_mat = join(input_path_this_version, "confusion_matrix_over_domains.csv")
                elif self.eval_mode == "single_domain":
                    file_confusion_mat = join(input_path_this_version, "confusion_matrix_over_sets.csv")
            # if use post-processed predictions, suffix the file name
            if self.use_post_processed_predictions:
                file_confusion_mat = file_confusion_mat.replace(".csv", "_post_processed.csv")
            df_confusion_mat = pd.read_csv(file_confusion_mat)
            # drop `Unnamed: 0` if exists
            if "Unnamed: 0" in df_confusion_mat.columns:
                df_confusion_mat = df_confusion_mat.drop(columns=["Unnamed: 0"])
            df_confusion_mat["version"] = version
            df_confusion_mat["task"] = task
            df_confusion_mat_over_domains = pd.concat([df_confusion_mat_over_domains, df_confusion_mat], ignore_index=True)
        self.df_confusion_mat_over_domains = df_confusion_mat_over_domains
        # # ----------
        # # Read in the config yaml files
        # # TODO
        # # ----------
        # # read in all the config over all versions of this task
        # list_file_config = glob(join(path_input_root, "*", task, "config.yaml"))
        # df_config = pd.DataFrame()
        # for file_config in list_file_config:
        #     with open(file_config, "r") as f:
        #         config = safe_load(f)
        #     df_config = pd.concat([df_config, pd.DataFrame(config, index=[0])], ignore_index=True)
        # self.df_config = df_config

    def visualize_metrics_over_domains_sep_metric(self, list_metrics, list_order_x_axis=None):
        """Visualize the evaluation results of the each domain. Visualize each version_output and each metric separately.
        Args:
            list_metrics (list(str)): e.g., ["accuracy","f1"]
            list_order_x_axis (list(str)): the order of the x-axis (e.g., ["source_train_raw","source_vali_raw","source_train_upsampled","source_vali_upsampled",...]). This also filters the data.
        """
        path_output = self.path_output
        # create the directory if not exists
        if not exists(path_output):
            makedirs(path_output)

        # ----------
        # Reorder the "time_domain" in the df
        # ----------
        if list_order_x_axis:
            if self.eval_mode == "gda":
                x_axis_key = "time_domain"
            elif self.eval_mode == "single_domain":
                x_axis_key = "set"
            # filter the data
            self.df_metrics_over_domains = self.df_metrics_over_domains[self.df_metrics_over_domains[x_axis_key].isin(list_order_x_axis)]
            # reorder the "time_domain" in the df
            self.df_metrics_over_domains[x_axis_key] = self.df_metrics_over_domains[x_axis_key].astype("category")
            self.df_metrics_over_domains[x_axis_key] = \
                self.df_metrics_over_domains[x_axis_key].cat.set_categories(list_order_x_axis)

        # ----------
        # Loop over each version
        # ----------
        for version in self.list_version_output:
            df_metrics_over_domains_version = self.df_metrics_over_domains[self.df_metrics_over_domains["version"] == version]

            # ----------
            # Loop over each metric
            # - one figure for each metric
            # ----------
            for metric in list_metrics:
                # ----------
                # Visualize the metrics over domains
                # ----------
                plt.figure()
                # create the line plot
                if self.model_type in ["gda_baseline",
                                       "gda_pseudo_cumulative",
                                       "gda_pseudo_buffer",
                                       "gda_pseudo_all"]:
                    ax = sns.lineplot(data=df_metrics_over_domains_version,
                                      x="time_domain",
                                      y=metric,
                                      markers="o",
                                      color="black")
                elif self.model_type == "single_domain_baseline":
                    # add a new column "sampled_type" (raw, upsampled/downsampled)
                    df_metrics_over_domains_version["sampled_type"] = df_metrics_over_domains_version["set"].str.extract("(raw|upsampled|downsampled)")
                    df_metrics_over_domains_version["sampled_type"] = \
                        df_metrics_over_domains_version["sampled_type"].astype("category")
                    df_metrics_over_domains_version['sampled_type'] = \
                        df_metrics_over_domains_version['sampled_type'].cat.set_categories(["raw", "upsampled", "downsampled"])
                    df_metrics_over_domains_version['sampled_type'] =\
                        df_metrics_over_domains_version['sampled_type'].cat.reorder_categories(["raw", "upsampled", "downsampled"])
                    df_metrics_over_domains_version['sampled_type'] =\
                        df_metrics_over_domains_version['sampled_type'].cat.remove_unused_categories()

                    # add a new column "set_type" (train, vali, test)
                    df_metrics_over_domains_version["set_type"] = df_metrics_over_domains_version["set"].str.extract("(train|vali|test)")
                    # - order the levels
                    df_metrics_over_domains_version["set_type"] = \
                        df_metrics_over_domains_version["set_type"].astype("category")
                    df_metrics_over_domains_version['set_type'] = df_metrics_over_domains_version['set_type'].cat.reorder_categories(["train", "vali", "test"])

                    ax = sns.lineplot(data=df_metrics_over_domains_version,
                                      x="set_type",
                                      y=metric,
                                      markers=["o", "o"],
                                      hue="sampled_type",
                                      palette=['b', 'r'],
                                      style="sampled_type",
                                      dashes=["", (2, 2)])
                elif self.model_type.startswith("llm_"):
                    if self.eval_mode == "gda":
                        ax = sns.lineplot(data=df_metrics_over_domains_version,
                                          x="time_domain",
                                          y=metric,
                                          markers="o",
                                          color="black")
                    elif self.eval_mode == "single_domain":
                        # add a new column "set_type" (train, vali, test)
                        df_metrics_over_domains_version["set_type"] = df_metrics_over_domains_version["set"].str.extract("(train|vali|test)")
                        # - order the levels
                        df_metrics_over_domains_version["set_type"] = \
                            df_metrics_over_domains_version["set_type"].astype("category")
                        df_metrics_over_domains_version['set_type'] = df_metrics_over_domains_version['set_type'].cat.reorder_categories(["train", "vali", "test"])
                        ax = sns.lineplot(data=df_metrics_over_domains_version,
                                          x="set",
                                          y=metric,
                                          markers="o",
                                          color="black")
                # rotate the x-axis labels to avoid overlapping
                plt.xticks(rotation=40, ha="right")

                # add text labels for the points
                for line in ax.lines:
                    for dot in line.get_xydata():
                        x, y = dot
                        ax.text(x, y + 0.05, f"{y:.2f}", ha="left", va="center", color=line.get_color())

                # set the range of y-axis
                ax.set(ylim=(0, 1))

                plt.tight_layout()
                # save the plot
                file_plot = join(path_output, f"{version}_{metric}.png")
                plt.savefig(file_plot)
                # close the plot to avoid overlaying
                plt.clf()

    def visualize_metrics_over_domains_comb_metrics(self, list_metrics, list_order_x_axis=None):
        """Visualize the evaluation results of the each domain. Visualize each version_output separately, with all metrics combined in one figure.
        Args:
            list_metrics (list(str)): e.g., ["accuracy","f1"]
            list_order_x_axis (list(str)): the order of the x-axis (e.g., ["source_train_raw","source_vali_raw","source_train_upsampled","source_vali_upsampled",...]). This also filters the data.
        """
        path_output = self.path_output
        # create the directory if not exists
        if not exists(path_output):
            makedirs(path_output)

        if self.eval_mode == "gda":
            x_axis_key = "time_domain"
        elif self.eval_mode == "single_domain":
            x_axis_key = "set"

        # ----------
        # Reorder the "time_domain" in the df
        # ----------
        if list_order_x_axis:
            # filter the data
            self.df_metrics_over_domains = self.df_metrics_over_domains[self.df_metrics_over_domains[x_axis_key].isin(list_order_x_axis)]
            # reorder the "time_domain" in the df
            self.df_metrics_over_domains[x_axis_key] = self.df_metrics_over_domains[x_axis_key].astype("category")
            self.df_metrics_over_domains[x_axis_key] = \
                self.df_metrics_over_domains[x_axis_key].cat.set_categories(list_order_x_axis)

        # ----------
        # Loop over each version
        # ----------
        for version in self.list_version_output:
            df_metrics_over_domains_version = self.df_metrics_over_domains[self.df_metrics_over_domains["version"] == version]

            # ----------
            # Put all metrics into one figure
            # - after plotting the macro-level metrics, plot the by-class metrics
            # -- by-class metrics should start from a new row
            # ----------
            # - https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
            # - https://stackoverflow.com/questions/65647127/how-do-i-combine-these-two-line-plots-together-using-seaborn
            # get the number of macro-level metrics
            num_macro_metrics = len([metric for metric in list_metrics if "macro" in metric or metric == "accuracy"])
            num_cols = len(self.par.DICT_STANCES_CODE[self.dataset])

            i_row_macro_end = ceil(num_macro_metrics / num_cols)
            num_rows_macro = i_row_macro_end
            num_rows_byclass = ceil((len(list_metrics) - num_macro_metrics) / num_cols)

            num_rows = num_rows_macro + num_rows_byclass
            fig_multi, axes_multi = plt.subplots(ncols=num_cols,
                                                 nrows=num_rows,
                                                 figsize=(15, 15 * (num_rows / 5)))

            # ----------
            # Loop over each metric
            # ----------
            for i_metric, metric in enumerate(list_metrics):
                # get the index of row and column
                # - macro-level metrics
                if i_metric < num_macro_metrics:
                    i_row = list_metrics.index(metric) // num_cols
                    i_col = list_metrics.index(metric) % num_cols
                else:
                    i_row = i_row_macro_end + (list_metrics.index(metric) - num_macro_metrics) // num_cols
                    i_col = (list_metrics.index(metric) - num_macro_metrics) % num_cols
                # ----------
                # Visualize the metrics over domains
                # ----------
                # create the line plot
                if self.model_type in ["gda_baseline",
                                       "gda_pseudo_cumulative",
                                       "gda_pseudo_buffer",
                                       "gda_pseudo_all"]:
                    ax = sns.lineplot(data=df_metrics_over_domains_version,
                                      x="time_domain",
                                      y=metric,
                                      markers="o",
                                      color="black",
                                      ax=axes_multi[i_row][i_col],
                                      label=metric)
                    # hide the x-axis ticks except the last row
                    if i_row != num_rows - 1:
                        ax.set(xticklabels=[])
                    else:
                        # rotate the x-axis labels to avoid overlapping
                        # plt.xticks(rotation=40, ha="right")
                        ax.tick_params('x', labelrotation=90)
                elif self.model_type == "single_domain_baseline":
                    if self.eval_mode == "single_domain":
                        if "sampled_type" not in df_metrics_over_domains_version.columns:
                            # add a new column "sampled_type" (raw, upsampled, downsampled)
                            df_metrics_over_domains_version["sampled_type"] = df_metrics_over_domains_version["set"].str.extract("(raw|upsampled|downsampled)")
                            df_metrics_over_domains_version["sampled_type"] = \
                                df_metrics_over_domains_version["sampled_type"].astype("category")
                            df_metrics_over_domains_version['sampled_type'] = \
                                df_metrics_over_domains_version['sampled_type'].cat.set_categories(["raw", "upsampled", "downsampled"])
                            df_metrics_over_domains_version['sampled_type'] = \
                                df_metrics_over_domains_version['sampled_type'].cat.reorder_categories(["raw", "upsampled", "downsampled"])
                            df_metrics_over_domains_version['sampled_type'] =\
                                df_metrics_over_domains_version['sampled_type'].cat.remove_unused_categories()
                            # add a new column "set_type" (train, vali, test)
                            df_metrics_over_domains_version["set_type"] = df_metrics_over_domains_version["set"].str.extract("(train|vali|test)")
                            # - order the levels
                            df_metrics_over_domains_version["set_type"] = \
                                df_metrics_over_domains_version["set_type"].astype("category")
                            df_metrics_over_domains_version['set_type'] = df_metrics_over_domains_version['set_type'].cat.reorder_categories(["train", "vali", "test"])

                        ax = sns.lineplot(data=df_metrics_over_domains_version,
                                          x="set_type",
                                          y=metric,
                                          markers=["o", "o"],
                                          hue="sampled_type",
                                          palette=['b', 'r'],
                                          style="sampled_type",
                                          dashes=["", (2, 2)],
                                          ax=axes_multi[i_row][i_col]
                                          )
                    elif self.eval_mode == "gda":
                        ax = sns.lineplot(data=df_metrics_over_domains_version,
                                          x=x_axis_key,
                                          y=metric,
                                          markers="o",
                                          color="black",
                                          ax=axes_multi[i_row][i_col],
                                          label=metric)
                        # hide the x-axis ticks except the last row
                        if i_row != num_rows - 1:
                            ax.set(xticklabels=[])
                        else:
                            # rotate the x-axis labels to avoid overlapping
                            # plt.xticks(rotation=40, ha="right")
                            ax.tick_params('x', labelrotation=90)
                elif self.model_type.startswith("llm_"):
                    ax = sns.lineplot(data=df_metrics_over_domains_version,
                                      x=x_axis_key,
                                      y=metric,
                                      markers="o",
                                      color="black",
                                      ax=axes_multi[i_row][i_col],
                                      label=metric)
                    # hide the x-axis ticks except the last row
                    if i_row != num_rows - 1:
                        ax.set(xticklabels=[])
                    else:
                        # rotate the x-axis labels to avoid overlapping
                        # plt.xticks(rotation=40, ha="right")
                        ax.tick_params('x', labelrotation=90)

                if self.eval_mode == "gda":
                    font_size = 8
                elif self.eval_mode == "single_domain":
                    font_size = 10

                # add text labels for the points
                for line in ax.lines:
                    for dot in line.get_xydata():
                        x, y = dot
                        ax.text(x, y + 0.05, f"{y:.2f}",
                                ha="left", va="center",
                                color=line.get_color(),
                                size=font_size)

                # set the range of y-axis
                ax.set(ylim=(0, 1))
                # hide the x-axis label
                ax.set(xlabel=None)
                # plt.tight_layout()
            plt.tight_layout()
            # save the plot
            file_plot = join(path_output, f"{version}_comb_metrics.png")
            plt.savefig(file_plot)
            # close the plot to avoid overlaying
            plt.clf()

    def visualize_confusion_metrices_over_domains_comb(self, list_sets_confusion_mat, preserve_order_list_sets=True,return_plot=False):
        """Visualize the confusion metrics over domains and combine them into one plot.
        Args:
            list_sets_confusion_mat (list(str)): e.g., ["train_raw","vali_raw","test_raw"]
            preserve_order_list_sets (list(str)): Whether to preserve the order of the sets specified in list_sets_confusion_mat. If True, the order of the sets in the plot will be the same as the order in list_sets_confusion_mat.
        """
        path_output = self.path_output
        # create the directory if not exists
        if not exists(path_output):
            makedirs(path_output)
        # ----------
        # Loop over each version
        # ----------
        for version in self.list_version_output:
            df_confusion_mat_over_domains_version = self.df_confusion_mat_over_domains[self.df_confusion_mat_over_domains["version"] == version]
            df_confusion_mat_over_domains_version = df_confusion_mat_over_domains_version.set_index("true_class")
            # ----------
            # Put all confusion matrices into one figure
            # - columns:
            # -- raw matrix
            # -- row-normalized matrix
            # -- column-normalized matrix
            # - rows:
            # -- domains/sets
            # ----------
            # - https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
            # - https://stackoverflow.com/questions/65647127/how-do-i-combine-these-two-line-plots-together-using-seaborn
            num_cols = 3
            if preserve_order_list_sets:
                pass
            else:
                # set the order of the sets to the row order in the csv
                df_confusion_mat_over_domains_version = df_confusion_mat_over_domains_version[df_confusion_mat_over_domains_version["data_set"].isin(list_sets_confusion_mat)]
                if not all(df_confusion_mat_over_domains_version.data_set.unique() == list_sets_confusion_mat):
                    list_sets_confusion_mat = df_confusion_mat_over_domains_version.data_set.unique()
            num_rows = len(list_sets_confusion_mat)
            fig_multi, axes_multi = plt.subplots(ncols=num_cols,
                                                 nrows=num_rows,
                                                 figsize=(15, 15 * (num_rows / 6)))
            # the column and row names to index the confusion matrix
            col_name_pred = ["pred_" + key for key in self.par.DICT_STANCES_CODE[self.dataset].keys()]
            row_name_true = ["true_" + key for key in self.par.DICT_STANCES_CODE[self.dataset].keys()]
            # num_classes = len(self.par.DICT_STANCES_CODE[self.dataset].keys())
            # list_classes = list(self.par.DICT_STANCES_CODE[self.dataset].keys())
            for i_dataset, data_set in enumerate(list_sets_confusion_mat):
                # ----------
                # raw confusion matrix
                # - https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
                # ----------
                # - get the confusion matrix
                df_confusion_mat_raw = df_confusion_mat_over_domains_version[df_confusion_mat_over_domains_version["data_set"] == data_set]
                array_raw = df_confusion_mat_raw.loc[row_name_true, col_name_pred].to_numpy()
                df_array_raw = pd.DataFrame(array_raw, row_name_true, col_name_pred)

                ax = sns.heatmap(df_array_raw,
                                 annot=True,
                                 fmt=".0f",
                                 # font size
                                 annot_kws={"size": 12},
                                 ax=axes_multi[i_dataset][0])
                ax.set_title(f"{data_set} raw (#)")
                # for label size
                ax.tick_params('x', labelrotation=0)
                # ----------
                # row-normalized confusion matrix
                # ----------
                # - get the confusion matrix
                df_confusion_mat_row_norm = df_confusion_mat_over_domains_version[df_confusion_mat_over_domains_version["data_set"] == data_set]
                array_row_norm = df_confusion_mat_row_norm.loc[row_name_true, col_name_pred].to_numpy()
                # - row-normalize the confusion matrix
                # -- * 100 to convert to percentage
                array_row_norm = 100 * array_row_norm / array_row_norm.sum(axis=1, keepdims=True)
                df_array_row_norm = pd.DataFrame(array_row_norm, row_name_true, col_name_pred)

                ax = sns.heatmap(df_array_row_norm,
                                 annot=True,
                                 fmt=".0f",
                                 # font size
                                 annot_kws={"size": 12},
                                 ax=axes_multi[i_dataset][1])
                ax.set_title(f"{data_set} row-normalized (%)")
                # for label size
                ax.tick_params('x', labelrotation=0)
                # ----------
                # column-normalized confusion matrix
                # ----------
                # - get the confusion matrix
                df_confusion_mat_col_norm = df_confusion_mat_over_domains_version[df_confusion_mat_over_domains_version["data_set"] == data_set]
                array_col_norm = df_confusion_mat_col_norm.loc[row_name_true, col_name_pred].to_numpy()
                # - column-normalize the confusion matrix
                # -- * 100 to convert to percentage
                array_col_norm = 100 * array_col_norm / array_col_norm.sum(axis=0, keepdims=True)
                df_array_col_norm = pd.DataFrame(array_col_norm, row_name_true, col_name_pred)

                ax = sns.heatmap(df_array_col_norm,
                                 annot=True,
                                 fmt=".0f",
                                 # font size
                                 annot_kws={"size": 12},
                                 ax=axes_multi[i_dataset][2])
                ax.set_title(f"{data_set} col-normalized (%)")
                # for label size
                ax.tick_params('x', labelrotation=0)
            # - set the figure title
            # fig_multi.suptitle(f"Confusion matrix for {self.dataset} - {self.model_type} - {version}", fontsize=16)
            plt.tight_layout()
            # save the plot
            file_plot = f"{self.model_type}_{version}_comb_confusion_mat.png"
            # - remove "llm_" and "single_domain_baseline_" at the beginning of model_type
            file_plot = file_plot.replace("llm_", "")
            file_plot = file_plot.replace("single_domain_baseline_", "")
            file_plot = join(path_output, file_plot)

            plt.savefig(file_plot)
            # close the plot to avoid overlaying
            plt.clf()

    def append_sum_confusion_matrix(self, list_domains_regex, name_set_across_domain, overwrite_input=True, file_output=None):
        """Compute the confusion matrix summed across domains and append the result to existing result csv (e.g., `confusion_matrix_over_domains.csv`). This can be used to summarize the performance of the model trained in the GDA context. For example, if we train the model on 10 target domains, we can compute the confusion matrix of the model on the 10 target domains.

        Args:
            list_domains_regex (list): list of regex to match the domains of interest. For example, if we want to compute the average metrics on the domains that start with "target", we can use `list_domains_regex = ["^target.*"]`.
            name_set_across_domain (str): the name of the set to append to the existing result csv. For example, if we want to compute the confusion matrix of the model on all the target domains, we can use `name_set_across_domain = "target_all"`.
            overwrite_input (bool): whether to overwrite the existing result csv. By default, the existing result csv will be overwritten.
            file_output (str): path to save the output csv files. Only used when `overwrite_input` is `False`. By default, the output csv files will overwrite the existing result csv by appending the average metric to the end.
        """
        assert self.eval_mode == "gda"
        assert isinstance(list_metrics, list)
        assert isinstance(list_domains_regex, list)
        assert all([isinstance(metric, str) for metric in list_metrics])
        assert all([isinstance(domain, str) for domain in list_domains_regex])

        # ----------
        # Loop over each version
        # ----------
        for version in self.list_version_output:
            # By default, the output csv files will overwrite the existing result csv by appending to the end.
            if not overwrite_input:
                assert file_output is not None
            else:
                path_output = self._get_input_path_this_version(version)
                file_output = join(path_output, "confusion_matrix_over_domains.csv")
            # Specify the corresponding confusion matrix
            # - by version
            df_confusion_mat_over_domains_version = self.df_confusion_mat_over_domains[self.df_confusion_mat_over_domains["version"] == version]
            # - by domains of interest
            df_confusion_mat_over_domains_version_domains = df_confusion_mat_over_domains_version[df_confusion_mat_over_domains_version["data_set"].str.match("|".join(list_domains_regex))]

            # skip the version if it has already been appended
            # - check if the column already exists
            if name_set_across_domain in df_confusion_mat_over_domains_version["data_set"].to_list():
                continue
            # ----------
            # Compute the confusion matric over all domains
            # - sum
            # ----------
            # select the columns which contain "pred"
            # - sum across domains
            df_confusion_mat_to_append = \
                df_confusion_mat_over_domains_version_domains[
                    # pred
                    ["pred_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()] +
                    # true
                    ["true_class"]
                ].groupby(["true_class"]).sum()
            # reorder the rows of df_confusion_mat_to_append
            df_confusion_mat_to_append = df_confusion_mat_to_append.reindex(
                ["true_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()]
            ).reset_index()
            # add the column `data_set`
            df_confusion_mat_to_append["data_set"] = name_set_across_domain

            # ----------
            # Append to the existing `df_confusion_mat_over_domains_version`
            # ----------
            df_confusion_mat_over_domains_version = \
                pd.concat([df_confusion_mat_over_domains_version, df_confusion_mat_to_append], ignore_index=True)

            # drop unneed columns
            # - version and task if they exist
            if "version" in df_confusion_mat_over_domains_version.columns:
                df_confusion_mat_over_domains_version = df_confusion_mat_over_domains_version.drop(columns=["version"])
            if "task" in df_confusion_mat_over_domains_version.columns:
                df_confusion_mat_over_domains_version = df_confusion_mat_over_domains_version.drop(columns=["task"])
            # ----------
            # Write to csv
            # ----------
            df_confusion_mat_over_domains_version.to_csv(file_output, index=True)

    def append_average_metric(self, list_metrics, list_domains_regex, name_set_domain_across_all, list_average_domain_methods=["macro", "micro"], overwrite_input=True, file_output=None):
        """Compute the average metrics across domains and append the result to existing result csv (e.g., `metric_over_domains.csv`). This can be used to summarize the performance of the model trained in the GDA context. For example, if we train the model on 10 target domains, we can compute the average performance of the model on the 10 target domains. This can use either macro-averaging or micro-averaging.

        Note that this computation is based on the confusion matrices of each target domain (`self.df_confusion_mat_over_domains`). Therefore, the domains of interest must be present in the input confusion matrices, and the metrics must be derivable from the confusion matrices (e.g., AUC is not derivable from the confusion matrices).

        Also note that this function assumes that `append_sum_confusion_matrix()` has been called before, because the confusion matrix summed across domains is needed to compute the micro-average metrics.

        Args:
            list_metrics (list): list of metrics of interest, e.g., ["accuracy", "f1_macro", "f1_FAVOR", ...]
            list_domains_regex (list): list of regex to match the domains of interest. For example, if we want to compute the average metrics on the domains that start with "target", we can use `list_domains_regex = ["^target.*"]`.
            name_set_domain_across_all (str): the name of the set to append to the existing result csv. This is also the name used to append rows in the confusion matrix by `append_sum_confusion_matrix()`. For example, if we computed the confusion matrix of the model on all the target domains, and want to also compute the metrics for the entire target domain, we can use `name_set_across_domain = "target_all"`.
            list_average_domain_methods (list): list of averaging methods. By default, both macro-averaging and micro-averaging are used. Macro-averaging treats each domain equally, while micro-averaging treats all domains as a single domain and compute a metric based on it.
            file_output (str): path to save the output csv files. By default, the output csv files will overwrite the existing result csv by appending the average metric to the end.
            overwrite_input (bool): whether to overwrite the existing result csv. By default, the existing result csv will be overwritten.
            file_output (str): path to save the output csv files. Only used when `overwrite_input` is `False`. By default, the output csv files will overwrite the existing result csv by appending the average metric to the end.
        """
        assert self.eval_mode == "gda"
        assert isinstance(list_metrics, list)
        assert isinstance(list_domains_regex, list)
        assert isinstance(list_average_domain_methods, list)
        assert isinstance(name_set_domain_across_all, str)
        assert all([isinstance(metric, str) for metric in list_metrics])
        assert all([isinstance(domain, str) for domain in list_domains_regex])
        assert all([isinstance(average_method, str) for average_method in list_average_domain_methods])
        assert all([average_method in ["macro", "micro"] for average_method in list_average_domain_methods])
        for metric in list_metrics:
            assert any([metric_allowed in metric for metric_allowed in ["accuracy", "f1", "precision", "recall"]])
        # ----------
        # Loop over each version
        # ----------
        for version in self.list_version_output:
            # By default, the output csv files will overwrite the existing result csv by appending to the end.
            if not overwrite_input:
                assert file_output is not None
            else:
                path_output = self._get_input_path_this_version(version)
                file_output = join(path_output, "metric_over_domains.csv")

            # Specify the corresponding confusion matrix
            # - used by micro-averaging
            # - by version
            df_confusion_mat_over_domains_version = self.df_confusion_mat_over_domains[self.df_confusion_mat_over_domains["version"] == version]
            # - only select the over-domain confusion matrix
            df_confusion_mat_over_domains_version = df_confusion_mat_over_domains_version[df_confusion_mat_over_domains_version["data_set"].str.match(name_set_domain_across_all)]

            # Specify the metrics of each domain
            # - used by macro-averaging
            # - by version
            df_metrics_over_domains_version = self.df_metrics_over_domains[self.df_metrics_over_domains["version"] == version]
            # - by domains of interest
            df_metrics_over_domains_version_domains = df_metrics_over_domains_version[df_metrics_over_domains_version["time_domain"].str.match("|".join(list_domains_regex))]

           # skip the version if it has already been appended
            # - check if the column already exists
            if name_set_domain_across_all + "_" + list_average_domain_methods[0] in df_metrics_over_domains_version["time_domain"].to_list():
                continue
            # ----------
            # Compute the metrics across domains (based on the confusion matrices)
            # ----------
            # - initialize the datafram to append to the existing `df_confusion_mat_over_domains_version``
            df_metrics_to_append = pd.DataFrame()

            for avg_method in list_average_domain_methods:
                # ----------
                # macro average: average the metrics across months
                # - NOT across classes
                # - have to get the metric for each domain first
                # -- can get directly from `df_confusion_mat_over_domains_version` (assuming that `append_sum_confusion_matrix()` has been called before)
                # ----------
                if avg_method == "macro":
                    df_metrics_to_append_macro = pd.DataFrame()
                    # compute the macro-average metrics
                    df_metrics_to_append_macro = pd.concat(
                        [df_metrics_to_append_macro,
                            df_metrics_over_domains_version_domains[list_metrics].mean(axis=0).to_frame().T], axis=0
                    )
                    # # add the columns that contain "num"
                    # # - the count of true class and pred class
                    # df_metrics_to_append_macro = pd.concat(
                    #     [df_metrics_to_append_macro,
                    #         df_metrics_over_domains_version[df_metrics_over_domains_version.columns[df_metrics_over_domains_version.columns.str.contains("num")]].sum(axis=0).to_frame().T], axis=1
                    # )
                    # add other columns
                    # - time_domain
                    df_metrics_to_append_macro["time_domain"] = name_set_domain_across_all + "_" + avg_method

                    # collect the metrics
                    df_metrics_to_append = \
                        pd.concat([df_metrics_to_append, df_metrics_to_append_macro], axis=0)

                # ----------
                # micro average: average the metrics across months
                # - NOT across classes
                # -- can compute directly from `df_metrics_over_domains_version`
                # ----------
                elif avg_method == "micro":
                    df_metrics_to_append_micro = pd.DataFrame()
                    for metric in list_metrics:
                        # get the over-domain confusion matrix
                        df_metrics_over_domains_version_domains = \
                            df_confusion_mat_over_domains_version[df_confusion_mat_over_domains_version["data_set"] == name_set_domain_across_all]

                        np_confusion_mat_over_domains_version = \
                            df_metrics_over_domains_version_domains.set_index("true_class").loc[
                                # true
                                ["true_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()],
                                # pred
                                ["pred_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()]
                            ].to_numpy()

                        if metric == "accuracy":
                            metric_name = metric
                            df_metrics_to_append_micro[metric] =\
                                [compute_metric_from_confusion_matrix(np_confusion_mat_over_domains_version,
                                                                      metric_name)]
                        else:
                            # split the metric name
                            metric_name, metric_suffix = metric.split("_")
                            if metric_suffix in ["macro", "micro"]:
                                average_type = metric_suffix
                                df_metrics_to_append_micro[metric] =\
                                    [compute_metric_from_confusion_matrix(np_confusion_mat_over_domains_version,
                                                                          metric_name, average_type)]
                            else:
                                name_class = metric_suffix
                                ind_positive_class = self.par.DICT_STANCES_CODE[self.dataset][name_class]
                                df_metrics_to_append_micro[metric] =\
                                    [compute_metric_from_confusion_matrix(np_confusion_mat_over_domains_version,
                                                                          metric_name, average_type="by_class",
                                                                          ind_positive_class=ind_positive_class)]
                    # add other columns
                    # - time_domain
                    df_metrics_to_append_micro["time_domain"] = name_set_domain_across_all + "_" + avg_method

                    # collect the metrics
                    df_metrics_to_append = \
                        pd.concat([df_metrics_to_append, df_metrics_to_append_micro], axis=0)

            # ----------
            # Compute the sum of each class across domains (both true and predicted)
            # ----------
            # add the columns that contain "num"
            # - the count of true class and pred class
            df_metrics_to_append = pd.concat(
                [df_metrics_to_append,
                    df_metrics_over_domains_version[df_metrics_over_domains_version.columns[df_metrics_over_domains_version.columns.str.contains("num")]].sum(axis=0).to_frame().T], axis=1
            )
            # ----------
            # Round
            # ----------
            df_metrics_to_append = df_metrics_to_append.round(4)
            # ----------
            # Append to the existing `df_metrics_over_domains_version`
            # ----------
            df_metrics_over_domains_version = \
                pd.concat([df_metrics_over_domains_version, df_metrics_to_append], ignore_index=True)

            # drop unneed columns
            # - version and task if they exist
            if "version" in df_metrics_over_domains_version.columns:
                df_metrics_over_domains_version = df_metrics_over_domains_version.drop(columns=["version"])
            if "task" in df_metrics_over_domains_version.columns:
                df_metrics_over_domains_version = df_metrics_over_domains_version.drop(columns=["task"])
            # ----------
            # Write to csv
            # ----------
            df_metrics_over_domains_version.to_csv(file_output, index=True)

    def write_hightlight_metrics_to_summary_csv(self, list_metrics_highlight, list_sets_highlight, col_name_set, path_output = None):
        """Write the metrics to a summary csv file.

        Args:
            list_metrics_highlight (list(str)): list of metrics to highlight.
            list_sets_highlight (list(str)): list of sets to highlight.
            col_name_set (str): name of the column that contains the set name. e.g., "time_domain" for gda_baseline.
            path_output (str, optional): path to the output folder. If None, use the default path. Defaults to None.
        
        Returns:
            pd.DataFrame: the dataframe of the highlight metrics.
        """
        assert col_name_set in self.df_metrics_over_domains.columns, \
            "The column name of the set name is not in the metrics dataframe."
        assert isinstance(list_metrics_highlight, list), "The list of metrics to highlight is not a list."
        assert isinstance(list_sets_highlight, list), "The list of sets to highlight is not a list."
        if path_output is None:
            path_output = self.path_output
        file_summary_csv = join(path_output, "metrics_highlights.csv")

        # ----------
        # Load the metrics
        # ----------
        df_metrics_over_domains_version = self.df_metrics_over_domains
        # ----------
        # Filter the metrics
        # ----------
        # - by domains
        df_metrics_over_domains_version = df_metrics_over_domains_version[df_metrics_over_domains_version[col_name_set].isin(list_sets_highlight)]
        # - by metrics
        col_to_keep = ["task", "version", col_name_set] + list_metrics_highlight
        df_metrics_over_domains_version = df_metrics_over_domains_version[col_to_keep]

        # ----------
        # add other columns (prepend to the existing columns)
        # - dataset, model_type, version_data
        # ----------
        df_metrics_over_domains_version["dataset"] = self.dataset
        df_metrics_over_domains_version["model_type"] = self.model_type
        # - reorder the columns so that the dataset and model_type are the first two columns
        df_metrics_over_domains_version = df_metrics_over_domains_version[["dataset", "model_type"] + df_metrics_over_domains_version.columns[:-2].tolist()]

        # make sure the length of the output versions is as specified
        assert df_metrics_over_domains_version["version"].nunique() == len(self.list_version_output), \
            "The number of versions in the output dataframe is not correct."
        assert len(df_metrics_over_domains_version) == len(self.list_version_output) * len(list_sets_highlight), \
            "The number of rows in the output dataframe is not correct."
        # ----------
        # order the rows by the version
        # - with existing csv
        # ----------
        if exists(file_summary_csv):
            df_metrics_over_domains_version_existing = \
                pd.read_csv(file_summary_csv)
            # merge
            df_metrics_over_domains_version = pd.concat([df_metrics_over_domains_version_existing, df_metrics_over_domains_version], ignore_index=True)
            # if there are rows sharing the same version, retain the one in df_metrics_over_domains_version and drop the one in df_metrics_over_domains_version_existing

            df_metrics_over_domains_version = df_metrics_over_domains_version.drop_duplicates(subset=["dataset", "model_type", "task", col_name_set, "version"], keep="last")
        df_metrics_over_domains_version = df_metrics_over_domains_version.sort_values(by=["version"])

        # ----------
        # Overwrite the existing summary csv
        # - the procedure above will make sure that the content in the existing csv is preserved
        # ----------
        df_metrics_over_domains_version = df_metrics_over_domains_version.round(4)
        df_metrics_over_domains_version.to_csv(file_summary_csv, index=False)
        return df_metrics_over_domains_version

    def _read_highlight_metrics_from_summary_csv(self, col_name_set):
        """Read the metrics from the summary csv file.

        Args:
            list_metrics_highlight (list(str)): list of metrics to highlight.
            list_sets_highlight (list(str)): list of sets to highlight.
            col_name_set (str): name of the column that contains the set name. e.g., "time_domain" for gda_baseline.

        Returns:
            pd.DataFrame: a dataframe containing the metrics.
        """

        path_output = self.path_output
        file_summary_csv = join(path_output, "metrics_highlights.csv")
        # ----------
        # Load the metrics
        # ----------
        df_metrics_over_domains_version = pd.read_csv(file_summary_csv)
        assert col_name_set in self.df_metrics_over_domains.columns, \
            "The column name of the set name is not in the metrics dataframe."
        return df_metrics_over_domains_version

    def _get_default_path_input_root(self):
        """Get the default path to the input root.

        Returns:
            path_input_root (str): path to the input root.
        """
        if "COVID_VACCINE" in self.dataset:
            if self.model_type == "gda_baseline":
                path_input_root = self.par.PATH_RESULT_COVID_VACCINE_GDA_BASELINE
            elif self.model_type == "gda_pseudo_cumulative":
                path_input_root = self.par.PATH_RESULT_COVID_VACCINE_GDA_PSUEDOLABEL_CUMULATIVE
            elif self.model_type == "gda_pseudo_buffer":
                path_input_root = self.par.PATH_RESULT_COVID_VACCINE_GDA_PSUEDOLABEL_BUFFER
            elif self.model_type == "gda_pseudo_all":
                path_input_root = self.par.PATH_RESULT_COVID_VACCINE_GDA_PSUEDOLABEL_ALL
            elif self.model_type == "single_domain_baseline":
                path_input_root = self.par.PATH_RESULT_COVID_VACCINE_SINGLE_DOMAIN_BASELINE
            elif self.model_type.startswith("llm"):
                _, gpt_model_type = self.model_type.split("_", 1)
                path_input_root = join(self.par.PATH_RESULT_COVID_VACCINE, "llm", gpt_model_type)

        elif self.dataset == "WTWT":
            if self.model_type == "gda_baseline":
                path_input_root = self.par.PATH_RESULT_WTWT_GDA_BASELINE
            elif self.model_type == "gda_pseudo_cumulative":
                path_input_root = self.par.PATH_RESULT_WTWT_GDA_PSUEDOLABEL_CUMULATIVE
            elif self.model_type == "gda_pseudo_buffer":
                path_input_root = self.par.PATH_RESULT_WTWT_GDA_PSUEDOLABEL_BUFFER
            elif self.model_type == "gda_pseudo_all":
                path_input_root = self.par.PATH_RESULT_WTWT_GDA_PSUEDOLABEL_ALL
            elif self.model_type == "single_domain_baseline":
                path_input_root = self.par.PATH_RESULT_WTWT_SINGLE_DOMAIN_BASELINE
            elif self.model_type.startswith("llm"):
                _, gpt_model_type = self.model_type.split("_", 1)
                path_input_root = join(self.par.PATH_RESULT_WTWT, "llm", gpt_model_type)

        elif self.dataset == "SEM_EVAL":
            if self.model_type == "single_domain_baseline":
                path_input_root = self.par.PATH_RESULT_SEM_EVAL_SINGLE_DOMAIN_BASELINE
            elif self.model_type.startswith("llm"):
                _, gpt_model_type = self.model_type.split("_", 1)
                path_input_root = join(self.par.PATH_RESULT_SEM_EVAL, "llm", gpt_model_type)
        else:
            raise ValueError("The dataset is not supported.")

        return path_input_root

    def _get_default_path_output(self):
        """Get the default path to save the output.

        Returns:
            path_output (str): path to save the output.
        """
        if "COVID_VACCINE" in self.dataset:
            if self.model_type == "gda_baseline":
                path_output = join(self.par.PATH_RESULT_COVID_VACCINE_GDA_BASELINE, "summary", task, "metrics_over_domains")
            elif self.model_type == "gda_pseudo_cumulative":
                path_output = join(self.par.PATH_RESULT_COVID_VACCINE_GDA_PSUEDOLABEL_CUMULATIVE, "summary", task, "metrics_over_domains")
            elif self.model_type == "gda_pseudo_buffer":
                path_output = join(self.par.PATH_RESULT_COVID_VACCINE_GDA_PSUEDOLABEL_BUFFER, "summary", task, "metrics_over_domains")
            elif self.model_type == "gda_pseudo_all":
                path_output = join(self.par.PATH_RESULT_COVID_VACCINE_GDA_PSUEDOLABEL_ALL, "summary", task, "metrics_over_domains")
            elif self.model_type == "single_domain_baseline":
                if self.eval_mode == "gda":
                    path_output = join(self.par.PATH_RESULT_COVID_VACCINE_SINGLE_DOMAIN_BASELINE,
                                       "summary", task, "metrics_over_domains")
                elif self.eval_mode == "single_domain":
                    path_output = join(self.par.PATH_RESULT_COVID_VACCINE_SINGLE_DOMAIN_BASELINE,
                                       "summary", task, "metrics_over_sets")
            elif self.model_type.startswith("llm"):
                _, gpt_model_type = self.model_type.split("_", 1)
                if self.eval_mode == "gda":
                    path_output = join(self.par.PATH_RESULT_COVID_VACCINE, "llm", gpt_model_type,
                                       "summary", task, "metrics_over_domains")
                elif self.eval_mode == "single_domain":
                    path_output = join(self.par.PATH_RESULT_COVID_VACCINE, "llm", gpt_model_type,
                                       "summary", task, "metrics_over_sets")
        elif self.dataset == "WTWT":
            if self.model_type == "gda_baseline":
                path_output = join(self.par.PATH_RESULT_WTWT_GDA_BASELINE, "summary", "metrics_over_domains")
            elif self.model_type == "gda_pseudo_cumulative":
                path_output = join(self.par.PATH_RESULT_WTWT_GDA_PSUEDOLABEL_CUMULATIVE, "summary", "metrics_over_domains")
            elif self.model_type == "gda_pseudo_buffer":
                path_output = join(self.par.PATH_RESULT_WTWT_GDA_PSUEDOLABEL_BUFFER, "summary", "metrics_over_domains")
            elif self.model_type == "gda_pseudo_all":
                path_output = join(self.par.PATH_RESULT_WTWT_GDA_PSUEDOLABEL_ALL, "summary", "metrics_over_domains")
            elif self.model_type == "single_domain_baseline":
                if self.eval_mode == "gda":
                    path_output = join(self.par.PATH_RESULT_WTWT_SINGLE_DOMAIN_BASELINE,
                                       "summary", "metrics_over_domains")
                elif self.eval_mode == "single_domain":
                    path_output = join(self.par.PATH_RESULT_WTWT_SINGLE_DOMAIN_BASELINE,
                                       "summary", "metrics_over_sets")
            elif self.model_type.startswith("llm"):
                _, gpt_model_type = self.model_type.split("_", 1)
                if self.eval_mode == "gda":
                    path_output = join(self.par.PATH_RESULT_WTWT, "llm", gpt_model_type,
                                       "summary", "metrics_over_domains")
                elif self.eval_mode == "single_domain":
                    path_output = join(self.par.PATH_RESULT_WTWT, "llm", gpt_model_type,
                                       "summary", "metrics_over_sets")
        elif self.dataset == "SEM_EVAL":
            if self.model_type == "single_domain_baseline":
                if self.eval_mode == "single_domain":
                    path_output = join(self.par.PATH_RESULT_SEM_EVAL_SINGLE_DOMAIN_BASELINE,
                                       "summary", "metrics_over_sets")
            elif self.model_type.startswith("llm"):
                _, gpt_model_type = self.model_type.split("_", 1)
                if self.eval_mode == "single_domain":
                    path_output = join(self.par.PATH_RESULT_SEM_EVAL, "llm", gpt_model_type,
                                       "summary", "metrics_over_sets")
        else:
            raise ValueError("The dataset is not supported.")
        return path_output

    def _get_input_path_this_version(self, version):
        """Get the path to the input file of this version.

        Args:
            version (str): version of the output.
        """
        if "COVID_VACCINE" in self.dataset:
            return join(self.path_input_root, version, self.task)
        elif self.dataset == "WTWT":
            return join(self.path_input_root, version)
        elif self.dataset == "SEM_EVAL":
            return join(self.path_input_root, version)
        else:
            raise ValueError("The dataset is not supported.")

    def _specify_summary_parameters(self, eval_mode, is_llm, target_domain_mode="all"):
        """
        Specify parameters for summarizing results.

        Args:
            eval_mode: str, "gda" or "single_domain"
            is_llm: bool, whether the model of interest is LLM.
            target_domain_mode: str, "all" or "test". Which set in the target domain to evaluate on. Only used when eval_mode == "gda". Default: "all".
        Returns:
            list_order_x_axis (list(str)): list of sets to plot in the order of x-axis.
            list_sets_confusion_mat (list(str)): list of sets to plot in the confusion matrix.
            preserve_order_list_sets (bool): whether to preserve the order of the sets in the confusion matrix.
            list_sets_highlight (list(str)): list of sets to highlight.
            list_metrics_highlight (list(str)): list of metrics to highlight.
            col_name_set (str): name of the column that contains the set name. e.g., "time_domain" for gda_baseline.
            list_metrics (list(str)): list of metrics to plot.
            name_set_domain_across_all (str): the name of the set to append to the existing result csv. This is also the name used to append rows in the confusion matrix by `append_sum_confusion_matrix()`. For example, if we computed the confusion matrix of the model on all the target domains, and want to also compute the metrics for the entire target domain, we can use `name_set_across_domain = "target_all"`.
            path_output (str): path to the output folder, e.g., "results/covid_vaccine/gda_baseline/summary/Q2/metrics_over_domains".
            path_input_root (str): path to the root of the input dir., e.g., "results/covid_vaccine/gda_baseline". It's will recursively search for the input dir for each version.
        """
        assert eval_mode in ["gda", "single_domain"], "The evaluation mode is not supported."
        assert isinstance(is_llm, bool), "The is_llm is not a boolean."
        assert target_domain_mode in ["all", "test"], "The target_domain_mode is not supported."
        # ----------
        # Specify the eval_mode specific parameters
        # ----------
        if eval_mode == "gda":
            if is_llm:
                list_order_x_axis = \
                    ["source_vali_raw", "source_test_raw"] + \
                    ["target_" + convert_time_unit_into_name(time) + "_" + target_domain_mode for time in self.par.DICT_TIMES[self.par.TIME_DOMAIN_UNIT][dataset]
                        if time not in self.par.DICT_SOURCE_DOMAIN[self.par.TIME_DOMAIN_UNIT][dataset]]
                if "source_train_raw" in set(self.df_metrics_over_domains["time_domain"]):
                    list_order_x_axis.insert(0, "source_train_raw")
            else:
                list_order_x_axis = \
                    ["source_train_upsampled", "source_vali_upsampled", "source_test_upsampled",
                        "source_train_raw", "source_vali_raw", "source_test_raw"] + \
                    ["target_" + convert_time_unit_into_name(time) + "_" + target_domain_mode for time in self.par.DICT_TIMES[self.par.TIME_DOMAIN_UNIT][dataset]
                     if time not in self.par.DICT_SOURCE_DOMAIN[self.par.TIME_DOMAIN_UNIT][dataset]]
            list_sets_confusion_mat = list_order_x_axis
            preserve_order_list_sets = True
            list_sets_highlight = ["target_" + target_domain_mode + "_macro"]
            col_name_set = "time_domain"
            name_set_domain_across_all = "target_" + target_domain_mode

        elif eval_mode == "single_domain":
            if is_llm:
                list_sets_confusion_mat = [
                    # raw (exclude train_raw)
                    "vali_raw", "test_raw"
                ]
            else:
                list_sets_confusion_mat = [
                    # raw
                    "train_raw", "vali_raw", "test_raw",
                    # upsampled
                    "train_upsampled", "vali_upsampled", "test_upsampled"
                ]
            list_order_x_axis = None
            preserve_order_list_sets = False
            list_sets_highlight = ["vali_raw", "test_raw"]
            col_name_set = "set"
            name_set_domain_across_all = None

        # ----------
        # Specify task-specific parameters
        # ----------
        if "COVID_VACCINE" in self.dataset:
            if self.task == "Q1":
                raise NotImplementedError("Q1 not implemented yet")
            elif self.task == "Q2_v1":
                list_metrics = [
                    # macro
                    "accuracy", "f1_macro", "precision_macro", "recall_macro",
                    # by-class
                    "f1_FAVOR", "f1_NON-FAVOR",
                    "precision_FAVOR", "precision_NON-FAVOR",
                    "recall_FAVOR", "recall_NON-FAVOR"]
                list_metrics_highlight = ["f1_macro", "f1_FAVOR", "f1_NON-FAVOR"]
            elif self.task == "Q2_v2":
                list_metrics = [
                    # macro
                    "accuracy", "f1_macro", "precision_macro", "recall_macro",
                    # by-class
                    "f1_AGAINST", "f1_NON-AGAINST",
                    "precision_AGAINST", "precision_NON-AGAINST",
                    "recall_AGAINST", "recall_NON-AGAINST"]
                list_metrics_highlight = ["f1_macro", "f1_AGAINST", "f1_NON-AGAINST"]
            elif self.task == "Q2":
                list_metrics = [
                    # macro
                    "accuracy", "f1_macro", "precision_macro", "recall_macro",
                    # by-class
                    "f1_NONE", "f1_FAVOR", "f1_AGAINST",
                    "precision_NONE", "precision_FAVOR", "precision_AGAINST",
                    "recall_NONE", "recall_FAVOR", "recall_AGAINST"]
                list_metrics_highlight = ["f1_macro", "f1_NONE", "f1_FAVOR", "f1_AGAINST"]
        elif self.dataset == "WTWT":
            # macro + by-class
            list_metrics = \
                ["accuracy", "f1_macro", "precision_macro", "recall_macro"] +\
                ["f1_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()] +\
                ["precision_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()] +\
                ["recall_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()]
            list_metrics_highlight = ["f1_macro"] + ["f1_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()]
        elif self.dataset == "SEM_EVAL":
            # macro + by-class
            list_metrics = \
                ["accuracy", "f1_macro", "precision_macro", "recall_macro"] +\
                ["f1_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()] +\
                ["precision_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()] +\
                ["recall_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()]
            list_metrics_highlight = ["f1_macro"] + ["f1_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()]
        else:
            raise NotImplementedError("The dataset is not supported.")
        return list_order_x_axis, list_sets_confusion_mat, preserve_order_list_sets, list_sets_highlight, list_metrics_highlight, col_name_set, list_metrics, name_set_domain_across_all


class MultiModelTypeComparer:
    def __init__(self, dataset, task, dict_result_summarizer, list_model_types, eval_mode, path_output=None):
        """Compare the models across domains.

        Args:
            dataset (str): "COVID_VACCINE_Q1" or "COVID_VACCINE_Q2"
            task (str): e.g., "Q1", "Q2"
            dict_result_summarizer (dict): a dictionary of ResultSummarizer objects, with the key being the model type. e.g., {"gda_baseline": ResultSummarizer, "gda_pseudo_cumulative": ResultSummarizer}
            list_model_types (list(str)): e.g., ["gda_baseline", "gda_pseudo_cumulative"]. The order of the list will be the order of the lines in the plot.
            eval_mode (str): "gda" or "single_domain".
            path_output (str): path to save the output csv files, e.g., "results/covid_vaccine/across_model_types/gda/Q2/metrics_over_domains". If None, it will use the default path. Default: None.
        """
        assert isinstance(dict_result_summarizer, dict)
        assert isinstance(list_model_types, list)
        # check dict_result_summarizer.keys() is the same as list_model_types
        assert set(dict_result_summarizer.keys()) == set(list_model_types)
        assert len(list_model_types) > 1, "Must have more than one model type."
        for model_type in list_model_types:
            assert isinstance(dict_result_summarizer[model_type], ResultSummarizer)
            assert model_type in ["gda_baseline",
                                  "gda_pseudo_cumulative", "gda_pseudo_buffer", "gda_pseudo_all",
                                  "single_domain_baseline"] or model_type.startswith("llm_"), "Not implemented yet for other model types."
            assert len(dict_result_summarizer[model_type].list_version_output) == 1, "Not implemented yet for multiple versions."
            assert dict_result_summarizer[model_type].eval_mode == eval_mode, "The eval_mode of the ResultSummarizer is not the same as the input eval_mode."
        assert eval_mode in ["gda", "single_domain"]
        self.dataset = dataset
        self.task = task
        self.dict_result_summarizer = dict_result_summarizer
        self.list_model_types = list_model_types
        if path_output is None:
            path_output = self._get_default_path_output()
        self.path_output = path_output
        self.eval_mode = eval_mode

    def visualize_compare_models_across_domains(self, list_metrics, list_order_x_axis=None):
        """Compare the models across domains.

        Args:
            list_metrics (list): list of metrics of interest, e.g., ["accuracy", "f1_macro", "f1_FAVOR", ...]
            list_order_x_axis (list(str)): the order of the x-axis (e.g., ["source_train_raw","source_vali_raw","source_train_upsampled","source_vali_upsampled",...])
        """
        assert isinstance(list_metrics, list)
        assert isinstance(list_order_x_axis, list) or list_order_x_axis is None
        # ----------
        # Initialize the ResultSummarizer for each model type
        # ----------
        # create the directory if not exists
        path_output = self.path_output
        if not exists(path_output):
            makedirs(path_output)
        dict_result_summarizer = self.dict_result_summarizer
        list_model_types = self.list_model_types

        if self.eval_mode == "gda":
            x_axis_key = "time_domain"
        elif self.eval_mode == "single_domain":
            x_axis_key = "set"
        # ----------
        # Reorder the "time_domain" in the df
        # ----------
        if list_order_x_axis:
            # reorder the "time_domain" in the df
            for model_type in self.list_model_types:
                dict_result_summarizer[model_type].df_metrics_over_domains[x_axis_key] = dict_result_summarizer[model_type].df_metrics_over_domains[x_axis_key].astype("category")
                dict_result_summarizer[model_type].df_metrics_over_domains[x_axis_key] = \
                    dict_result_summarizer[model_type].df_metrics_over_domains[x_axis_key].cat.set_categories(list_order_x_axis)

        # ----------
        # Loop over the model type to collect the metrics
        # ----------
        df_metrics_over_domains_over_model_types = pd.DataFrame()

        for model_type in list_model_types:
            df_metrics_over_domains_over_this = dict_result_summarizer[model_type].df_metrics_over_domains
            df_metrics_over_domains_over_this["model_type"] = model_type
            df_metrics_over_domains_over_model_types = pd.concat([df_metrics_over_domains_over_model_types, df_metrics_over_domains_over_this], ignore_index=True)

        # ----------
        # Order the model types
        # ----------
        df_metrics_over_domains_over_model_types["model_type"] = df_metrics_over_domains_over_model_types["model_type"].astype("category")
        df_metrics_over_domains_over_model_types["model_type"] = df_metrics_over_domains_over_model_types["model_type"].cat.set_categories(list_model_types)

        # ----------
        # Put all metrics into one figure
        # - after plotting the macro-level metrics, plot the by-class metrics
        # -- by-class metrics should start from a new row
        # ----------
        # - https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
        # - https://stackoverflow.com/questions/65647127/how-do-i-combine-these-two-line-plots-together-using-seaborn
        # get the number of macro-level metrics
        num_macro_metrics = len([metric for metric in list_metrics if "macro" in metric or metric == "accuracy"])
        num_cols = len(self.par.DICT_STANCES_CODE[dataset])

        i_row_macro_end = ceil(num_macro_metrics / num_cols)
        num_rows_macro = i_row_macro_end
        num_rows_byclass = ceil((len(list_metrics) - num_macro_metrics) / num_cols)

        num_rows = num_rows_macro + num_rows_byclass
        fig_multi, axes_multi = plt.subplots(ncols=num_cols,
                                             nrows=num_rows,
                                             figsize=(15, 15 * (num_rows / 5)))

        # ----------
        # Loop over each metric
        # ----------
        for i_metric, metric in enumerate(list_metrics):
            # get the index of row and column
            # - macro-level metrics
            if i_metric < num_macro_metrics:
                i_row = list_metrics.index(metric) // num_cols
                i_col = list_metrics.index(metric) % num_cols
            else:
                i_row = i_row_macro_end + (list_metrics.index(metric) - num_macro_metrics) // num_cols
                i_col = (list_metrics.index(metric) - num_macro_metrics) % num_cols
            # ----------
            # Visualize the metrics over domains
            # ----------
            # create the line plot
            if list_model_types == ["gda_baseline", "gda_pseudo_cumulative"]:
                list_markers = ["o", "o"]
                list_palette = ['b', 'r']
                list_dashes = ["", (2, 2)]
            elif list_model_types == ["gda_baseline", "gda_pseudo_buffer"]:
                list_markers = ["o", "o"]
                list_palette = ['b', 'g']
                list_dashes = ["", (2, 2)]
            elif list_model_types == ["gda_baseline", "gda_pseudo_buffer", "single_domain_baseline"]:
                list_markers = ["o", "o"]
                list_palette = ['b', 'g']
                list_dashes = ["", (2, 2)]
            elif list_model_types == ["gda_baseline", "gda_pseudo_buffer", "gda_pseudo_cumulative", "single_domain_baseline"]:
                list_markers = ["o", "o", "o"]
                list_palette = ['b', 'g', "r"]
                list_dashes = ["", (2, 2), (2, 2)]
            elif list_model_types == ["gda_baseline", "gda_pseudo_buffer", "gda_pseudo_all", "single_domain_baseline"]:
                list_markers = ["o", "o", "o"]
                list_palette = ['b', 'g', 'black']
                list_dashes = ["", (2, 2), (2.5, 2.5)]
            elif list_model_types == ["gda_baseline", "gda_pseudo_buffer", "gda_pseudo_cumulative"]:
                list_markers = ["o", "o", "o"]
                list_palette = ['b', 'g', 'r']
                list_dashes = ["", (2, 2), (2, 2)]
            elif list_model_types == ["gda_baseline", "gda_pseudo_buffer", "gda_pseudo_cumulative", "gda_pseudo_all"]:
                list_markers = ["o", "o", "o", "o"]
                list_palette = ['b', 'g', 'r', "black"]
                list_dashes = ["", (2, 2), (2, 2), (2.5, 2.5)]
            elif list_model_types == ["gda_baseline", "gda_pseudo_buffer", "gda_pseudo_cumulative", "gda_pseudo_all", "single_domain_baseline"]:
                list_markers = ["o", "o", "o", "o"]
                list_palette = ['b', 'g', 'r', "black"]
                list_dashes = ["", (2, 2), (2, 2), (2.5, 2.5)]
            elif list_model_types[:-1] == ["gda_baseline", "gda_pseudo_buffer", "gda_pseudo_cumulative"] and list_model_types[-1].startswith("llm_"):
                list_markers = ["o", "o", "o", "^"]
                list_palette = ['b', 'g', 'r', "black"]
                list_dashes = ["", (2, 2), (2, 2), (1, 1)]
            elif list_model_types[:-1] == ["gda_baseline", "gda_pseudo_cumulative", "gda_pseudo_all"] and list_model_types[-1].startswith("llm_"):
                list_markers = ["o", "o", "o", "^"]
                list_palette = ['b', 'r', "darkviolet", "black"]
                list_dashes = ["", (2, 2), (2.5, 2.5), (1, 1)]
            elif list_model_types[:-1] == ["single_domain_baseline"] and list_model_types[-1].startswith("llm_"):
                list_markers = ["o", "^"]
                list_palette = ['b', "black"]
                list_dashes = ["", (1, 1)]
            elif list_model_types == ["single_domain_baseline",
                                      "llm_flan-t5-xxl",
                                      "llm_chatgpt_turbo_3_5"]:
                list_markers = ["o", "^", "^"]
                list_palette = ['b', 'black', 'r']
                list_dashes = ["", (1, 1), (1, 1)]
            else:
                raise NotImplementedError("Not implemented yet for other order of model types.")
            if "single_domain_baseline" in list_model_types and self.eval_mode == "gda":
                # overlay later with a transparent line
                # - remove it from the category list
                df_plot = df_metrics_over_domains_over_model_types[df_metrics_over_domains_over_model_types["model_type"] != "single_domain_baseline"].copy()
                mask = df_plot['model_type'] == 'single_domain_baseline'
                df_plot.loc[mask, 'model_type'] = None
                df_plot['model_type'] = df_plot['model_type'].cat.remove_unused_categories()

            else:
                df_plot = df_metrics_over_domains_over_model_types
            ax = sns.lineplot(data=df_plot,
                              x=x_axis_key,
                              y=metric,
                              markers=list_markers,
                              hue="model_type",
                              palette=list_palette,
                              style="model_type",
                              dashes=list_dashes,
                              ax=axes_multi[i_row][i_col]
                              #   markersize = 4,
                              #   linewidth = 1.3
                              # label=metric
                              )
            # overlay a transparent line for the single_domain_baseline
            if "single_domain_baseline" in list_model_types and self.eval_mode == "gda":
                df_plot_single_domain_baseline = df_metrics_over_domains_over_model_types[df_metrics_over_domains_over_model_types["model_type"] == "single_domain_baseline"].copy()
                # - remove others from the category list
                df_plot_single_domain_baseline['model_type'] = df_plot_single_domain_baseline['model_type'].cat.remove_unused_categories()
                sns.lineplot(
                    data=df_plot_single_domain_baseline,
                    x=x_axis_key,
                    y=metric,
                    hue="model_type",
                    palette=["b"],
                    style="model_type",
                    dashes=[""],
                    alpha=0.5,
                    ax=ax
                )

            # hide the x-axis ticks except the last row
            if i_row != num_rows - 1:
                ax.set(xticklabels=[])
            else:
                # rotate the x-axis labels to avoid overlapping
                # plt.xticks(rotation=40, ha="right")
                ax.tick_params('x', labelrotation=90)

            # if set(list_model_types) == set(["gda_baseline", "gda_pseudo_cumulative"]):
            font_size = 8
            # else:
            #     raise NotImplementedError("Not implemented yet for other model types.")

            # # add text labels for the points
            # for line in ax.lines:
            #     for dot in line.get_xydata():
            #         x, y = dot
            #         ax.text(x, y + 0.05, f"{y:.2f}",
            #                 ha="left", va="center",
            #                 color=line.get_color(),
            #                 size=font_size)

            # set the range of y-axis
            ax.set(ylim=(0, 1))
            # hide the x-axis label
            ax.set(xlabel=None)
            # plt.tight_layout()
        plt.tight_layout()
        # save the plot
        # - "{model_type_1}_{version_model_type_1}_{model_type_2}_{version_model_type_2}_..._metrics.png}"
        list_str_file_plot = []
        for model_type in list_model_types:
            list_str_file_plot.append(f"{model_type}_{dict_result_summarizer[model_type].list_version_output[0]}")
        file_plot = join(path_output, "_".join(list_str_file_plot) + "_metrics.png")

        plt.savefig(file_plot)
        # close the plot to avoid overlaying
        plt.clf()

    def write_hightlight_metrics_to_summary_csv(self, col_name_set):
        """Write the metrics to a summary csv file.

        Args:
            path_output (str): path to the output csv file
            col_name_set (str): name of the column that contains the set name. e.g., "time_domain" for gda_baseline.
        """
        # ----------
        # Initialize the ResultSummarizer for each model type
        # ----------
        path_output = self.path_output
        # create the directory if not exists
        if not exists(path_output):
            makedirs(path_output)
        dict_result_summarizer = self.dict_result_summarizer
        list_model_types = self.list_model_types

        # ----------
        # Concatenate the metrics over domains over model types
        # ----------
        df_metrics_over_domains_over_model_types = pd.DataFrame()
        for model_type in self.list_model_types:
            df_metrics_over_domains_over_this = dict_result_summarizer[model_type]._read_highlight_metrics_from_summary_csv(col_name_set)
            df_metrics_over_domains_over_this["model_type"] = model_type
            # filter by version
            # - if isin list_version_output, then keep
            # - if not, then drop
            df_metrics_over_domains_over_this = df_metrics_over_domains_over_this[df_metrics_over_domains_over_this["version"].isin(dict_result_summarizer[model_type].list_version_output)]
            df_metrics_over_domains_over_model_types = pd.concat([df_metrics_over_domains_over_model_types, df_metrics_over_domains_over_this], ignore_index=True)

        assert col_name_set in df_metrics_over_domains_over_model_types.columns, \
            "The column name of the set name is not in the metrics dataframe."
        # check if all model type is present exactly once
        assert len(df_metrics_over_domains_over_model_types["model_type"].unique()) == len(list_model_types), \
            "The number of model types in the metrics dataframe is not equal to the number of model types."

        # ----------
        # Overwrite the existing summary csv
        # - the procedure above will make sure that the content in the existing csv is preserved
        # ----------
        # save the file
        # - "{model_type_1}_{version_model_type_1}_{model_type_2}_{version_model_type_2}_..._metrics.png}"
        list_str_file_plot = []
        for model_type in list_model_types:
            list_str_file_plot.append(f"{model_type}_{dict_result_summarizer[model_type].list_version_output[0]}")
        file_summary_csv = join(path_output, "_".join(list_str_file_plot) + "_metrics_highlights.csv")
        df_metrics_over_domains_over_model_types.to_csv(file_summary_csv, index=False)

    def _has_llm(self):
        """Check if the list of model types has LLM.

        Returns:
            bool: whether the list of model types has LLM.
        """
        list_model_types = self.list_model_types
        for model_type in list_model_types:
            if "llm" in model_type.lower():
                return True
        return False

    def _specify_summary_parameters(self, eval_mode, target_domain_mode="all"):
        """
        Specify parameters for summarizing results.

        Args:
            eval_mode: str, "gda" or "single_domain"
            target_domain_mode: str, "all" or "test". Which set in the target domain to evaluate on. Only used when eval_mode == "gda". Default: "all".
        Returns:
            list_order_x_axis (list(str)): list of sets to plot in the order of x-axis.
            list_sets_confusion_mat (list(str)): list of sets to plot in the confusion matrix.
            preserve_order_list_sets (bool): whether to preserve the order of the sets in the confusion matrix.
            col_name_set (str): name of the column that contains the set name. e.g., "time_domain" for gda_baseline.
            list_metrics (list(str)): list of metrics to plot.
            name_set_domain_across_all (str): the name of the set to append to the existing result csv. This is also the name used to append rows in the confusion matrix by `append_sum_confusion_matrix()`. For example, if we computed the confusion matrix of the model on all the target domains, and want to also compute the metrics for the entire target domain, we can use `name_set_across_domain = "target_all"`.
        """
        assert eval_mode in ["gda", "single_domain"], "eval_mode must be either 'gda' or 'single_domain'."
        assert target_domain_mode in ["all", "test"], "target_domain_mode must be either 'all' or 'test'."
        # ----------
        # Specify the eval_mode specific parameters
        # ----------
        if eval_mode == "gda":
            if self._has_llm():
                list_order_x_axis = \
                    ["source_vali_raw", "source_test_raw"] + \
                    ["target_" + convert_time_unit_into_name(time) + "_" + target_domain_mode for time in self.par.DICT_TIMES[self.par.TIME_DOMAIN_UNIT][dataset]
                     if time not in self.par.DICT_SOURCE_DOMAIN[self.par.TIME_DOMAIN_UNIT][dataset]]
            else:
                list_order_x_axis =\
                    ["source_train_raw", "source_vali_raw", "source_test_raw"] + \
                    ["target_" + convert_time_unit_into_name(time) + "_" + target_domain_mode for time in self.par.DICT_TIMES[self.par.TIME_DOMAIN_UNIT][dataset]
                     if time not in self.par.DICT_SOURCE_DOMAIN[self.par.TIME_DOMAIN_UNIT][dataset]]
            col_name_set = "time_domain"
            name_set_domain_across_all = "target_" + target_domain_mode
        elif eval_mode == "single_domain":
            if self._has_llm():
                list_order_x_axis = [
                    # raw (exclude train_raw)
                    "vali_raw", "test_raw"
                ]
            else:
                list_order_x_axis = ["train_raw", "vali_raw", "test_raw"]
            col_name_set = "set"
            name_set_domain_across_all = None

        # ----------
        # Specify task-specific parameters
        # ----------
        if "COVID_VACCINE" in self.dataset:
            if self.task == "Q1":
                raise NotImplementedError("Q1 not implemented yet")
            elif self.task == "Q2_v1":
                list_metrics = [
                    # macro
                    "accuracy", "f1_macro", "precision_macro", "recall_macro",
                    # by-class
                    "f1_FAVOR", "f1_NON-FAVOR",
                    "precision_FAVOR", "precision_NON-FAVOR",
                    "recall_FAVOR", "recall_NON-FAVOR"]
            elif self.task == "Q2_v2":
                list_metrics = [
                    # macro
                    "accuracy", "f1_macro", "precision_macro", "recall_macro",
                    # by-class
                    "f1_AGAINST", "f1_NON-AGAINST",
                    "precision_AGAINST", "precision_NON-AGAINST",
                    "recall_AGAINST", "recall_NON-AGAINST"]
            elif self.task == "Q2":
                list_metrics = [
                    # macro
                    "accuracy", "f1_macro", "precision_macro", "recall_macro",
                    # by-class
                    "f1_NONE", "f1_FAVOR", "f1_AGAINST",
                    "precision_NONE", "precision_FAVOR", "precision_AGAINST",
                    "recall_NONE", "recall_FAVOR", "recall_AGAINST"]
        elif self.dataset == "WTWT":
            # macro + by-class
            list_metrics = \
                ["accuracy", "f1_macro", "precision_macro", "recall_macro"] +\
                ["f1_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()] +\
                ["precision_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()] +\
                ["recall_" + label for label in self.par.DICT_STANCES_CODE[self.dataset].keys()]
        list_sets_confusion_mat = list_order_x_axis
        preserve_order_list_sets = True

        return list_order_x_axis, list_sets_confusion_mat, preserve_order_list_sets, col_name_set, list_metrics, name_set_domain_across_all

    def _get_default_path_output(self):
        """Get the default path to save the output.

        Returns:
            path_output (str): path to save the output.
        """
        if "COVID_VACCINE" in self.dataset:
            if eval_mode == "gda":
                path_output = join(self.par.PATH_RESULT_COVID_VACCINE_ACROSS_MODEL_TYPES_GDA, task, "metrics_over_domains")
            elif eval_mode == "single_domain":
                path_output = join(self.par.PATH_RESULT_COVID_VACCINE_ACROSS_MODEL_TYPES_SINGLE_DOMAIN, task, "metrics_over_sets")

        elif self.dataset == "WTWT":
            if eval_mode == "gda":
                path_output = join(self.par.PATH_RESULT_WTWT_ACROSS_MODEL_TYPES_GDA, "metrics_over_domains")
            elif eval_mode == "single_domain":
                path_output = join(self.par.PATH_RESULT_WTWT_ACROSS_MODEL_TYPES_SINGLE_DOMAIN, "metrics_over_sets")
        return path_output


if __name__ == "__main__":
    par = get_parameters_for_dataset()
    # run the summarizer
    parser = argparse.ArgumentParser(description='Result Summarizer.')

    # - COVID_VACCINE
    # DATASET = "COVID_VACCINE"
    # TASK = "Q2"
    # TASK = "Q2_v1"
    # TASK = "Q2_v2"

    # - WTWT
    # DATASET = "WTWT"
    # TASK = None

    # - SemEval
    DATASET = "SEM_EVAL"
    TASK = None

    # DATASET = "COVID_VACCINE_" + TASK
    # TARGET_DOMAIN_MODE = "all"
    TARGET_DOMAIN_MODE = "test"
    # - for a single model type
    MODEL_TYPE = "single_domain_baseline"
    # MODEL_TYPE = "gda_baseline"
    # MODEL_TYPE = "gda_pseudo_cumulative"
    # MODEL_TYPE = "gda_pseudo_buffer"
    # MODEL_TYPE = "gda_pseudo_all"
    # MODEL_TYPE = "llm_flan-t5-xxl"
    MODEL_TYPE = "llm_chatgpt_turbo_3_5"

    # - for comparing multiple model types
    # MODEL_TYPE = ["gda_baseline", "gda_pseudo_buffer"]
    # MODEL_TYPE = ["gda_baseline", "gda_pseudo_buffer", "single_domain_baseline"]
    # MODEL_TYPE = ["gda_baseline", "gda_pseudo_buffer", "gda_pseudo_cumulative", "single_domain_baseline"]
    # MODEL_TYPE = ["gda_baseline", "gda_pseudo_buffer", "gda_pseudo_all", "single_domain_baseline"]
    # MODEL_TYPE = ["gda_baseline", "gda_pseudo_cumulative"]
    # MODEL_TYPE = ["gda_baseline", "gda_pseudo_buffer", "gda_pseudo_cumulative"]
    # MODEL_TYPE = ["gda_baseline", "gda_pseudo_buffer", "gda_pseudo_cumulative", "gda_pseudo_all"]
    # MODEL_TYPE = ["gda_baseline", "gda_pseudo_buffer", "gda_pseudo_cumulative", "gda_pseudo_all", "single_domain_baseline"]
    # MODEL_TYPE = ["gda_baseline", "gda_pseudo_cumulative", "gda_pseudo_all", "llm_flan-t5-xxl"]
    # MODEL_TYPE = ["gda_baseline", "gda_pseudo_cumulative", "gda_pseudo_all", "llm_chatgpt_turbo_3_5"]

    # MODEL_TYPE = ["single_domain_baseline", "llm_flan-t5-xxl"]
    # MODEL_TYPE = ["single_domain_baseline", "llm_flan-t5-xxl", "llm_chatgpt_turbo_3_5"]

    # whether to use the post-processed predictions (e.g., v9 in single_domain, flan-t5-xxl)
    USE_POST_PROCESSED_PREDICTIONS = False

    # - evaluation mode ("gda" or "single_domain_baseline")
    # -- this will be overwritten if the model type is "gda_*"
    # -- only used for LLM results and "single_domain_baseline"
    # EVAL_MODE = "gda"
    EVAL_MODE = "single_domain"

    parser.add_argument('--task', default=TASK, help='Specify the task')
    parser.add_argument('--dataset', default=DATASET, help='Specify the dataset')
    parser.add_argument('--model_type', nargs='*', default=MODEL_TYPE, help='Specify the model type.')
    parser.add_argument('--target_domain_mode', default=TARGET_DOMAIN_MODE, help='Specify the target domain mode. Either "all" or "test"')
    parser.add_argument(
        '--eval_mode',
        default=EVAL_MODE,
        help='Specify the evaluation mode. "gda" or "single_domain_baseline". This will be overwritten if any model type is "gda_*" because "gda" is the only reasonable evaluation mode for GDA models.')
    parser.add_argument('--use_post_processed_predictions', default=USE_POST_PROCESSED_PREDICTIONS,
                        help='Specify whether to use the post-processed predictions (e.g., v9 in single_domain, flan-t5-xxl)')

    # parser.add_argument('--use_newest_version_output', nargs='*', default=, help='Specify the model type')
    args = parser.parse_args()
    task = args.task
    dataset = args.dataset
    model_type = parse_list_to_list_or_str(args.model_type)
    # - get gpt model types from model_type, e.g., "llm_flan-t5-xxl" -> "flan-t5-xxl"
    # - extract the substring after the last "llm_"
    list_gpt_model_types = [m.split("llm_")[-1] for m in model_type if "llm_" in m]
    target_domain_mode = args.target_domain_mode
    eval_mode = args.eval_mode
    use_post_processed_predictions = args.use_post_processed_predictions

    if isinstance(model_type, str):
        if model_type.startswith("gda"):
            eval_mode = "gda"
    elif isinstance(model_type, list):
        for m in model_type:
            if m.startswith("gda"):
                eval_mode = "gda"
                break
    if dataset not in ["COVID_VACCINE", "WTWT"]:
        eval_mode = "single_domain"

    # ----------
    # single model type
    # ----------
    if isinstance(model_type, str):
        # ----------
        # gda_baseline
        # ----------
        if model_type == "gda_baseline":
            # COVID_VACCINE
            # # list_versions = \
            # #     ["v" + str(v) for v in range(15, 16)] + ["v" + str(v) for v in range(17, 23)]
            # # list_versions = ["v21"]
            # list_versions = ["v23_Q2_v2"]

            # WTWT
            list_versions = ["v1"]

        # ----------
        # gda_pseudo_cumulative
        # ----------
        elif model_type == "gda_pseudo_cumulative":
            # COVID_VACCINE
            # list_versions = ["v1", "v2", "v3"]
            # list_versions = ["v8_Q2_v2"]
            # list_versions = ["vtest"]

            # WTWT
            list_versions = ["v1"]
        # ----------
        # gda_pseudo_buffer
        # ----------
        elif model_type == "gda_pseudo_buffer":
            # COVID_VACCINE
            # list_versions = ["v" + str(v) for v in range(1, 5)]
            # list_versions = ["v3"]
            # list_versions = ["v5_Q2_v2"]

            # WTWT
            list_versions = ["v1"]
        # ----------
        # gda_pseudo_all
        # ----------
        elif model_type == "gda_pseudo_all":
            # COVID_VACCINE
            # list_versions = ["v" + str(v) for v in range(1, 5)]
            # list_versions = ["v1"]
            # list_versions = ["v2_Q2_v2"]

            # WTWT
            list_versions = ["v1"]
        # ----------
        # single_domain_baseline
        # ----------
        elif model_type == "single_domain_baseline":
            # COVID_VACCINE
            # list_versions = ["v1", "v2", "v3", "v4", "v5"]
            # list_versions = ["v16"]
            # list_versions = ["v15_Q2_v1"]
            # list_versions = ["v15_Q2_v1"]
            # WTWT
            # list_versions = ["v1"]
            # SEMEVAL
            list_versions = ["v3"]
        elif model_type.startswith("llm"):
            # split the string
            # llm_gpt_3_davinci -> llm, gpt_3_davinci
            _, gpt_model_type = model_type.split("_", 1)

            if gpt_model_type == "gpt_3_davinci":
                # # - gpt_3_davinci
                list_versions = ["v2"]
                # list_versions = ["v2_Q2_v1"]
            elif gpt_model_type == "flan-t5-xxl":
                # COVID_VACCINE
                # # - flan-t5-xxl
                # list_versions = ["v10"]
                # list_versions = ["v9_post_processed"]

                # SEMEVAL
                list_versions = ["v2"]
            elif gpt_model_type == "flan_ul2":
                # - flan_ul2
                list_versions = ["v1"]
            elif gpt_model_type == "chatgpt_turbo_3_5":
                # COVID_VACCINE                
                # - chatgpt_turbo_3_5
                # list_versions = ["v1"]

                # SEMEVAL
                list_versions = ["v3"]

    elif isinstance(model_type, list):
        if eval_mode == "gda":
            # COVID_VACCINE
            # dict_versions = \
            #     {"gda_baseline": ["v23_Q2_v1"],
            #         "gda_pseudo_buffer": ["v5_Q2_v1"],
            #         "gda_pseudo_cumulative": ["v8_Q2_v1"],
            #         "gda_pseudo_all": ["v2_Q2_v1"],
            #         # "llm_flan-t5-xxl": ["v8"],
            #         # # --- gpt-chat-turbo-3_5
            #         # "llm_chatgpt_turbo_3_5": ["v1"]
            #         "single_domain_baseline": ["v15_Q2_v1"]
            #      }
            # WTWT
            dict_versions = \
                {"gda_baseline": ["v1"],
                 "gda_pseudo_buffer": ["v1"],
                 "gda_pseudo_cumulative": ["v1"],
                 "gda_pseudo_all": ["v1"],
                 "single_domain_baseline": ["v1"]
                 }
        elif eval_mode == "single_domain":
            dict_versions = \
                {"single_domain_baseline": ["v14"],
                 "llm_flan-t5-xxl": ["v9"],
                 "llm_chatgpt_turbo_3_5": ["v1"]
                 }

    # ----------
    # Initialize the summarizer
    # ----------
    # ----------
    # single model type
    # ----------
    if isinstance(model_type, str):
        summarizer = ResultSummarizer(dataset, list_versions, task, model_type, eval_mode, use_post_processed_predictions=use_post_processed_predictions)
        is_llm = model_type.startswith("llm_")

        # get the parameters for summarizing the results
        list_order_x_axis, list_sets_confusion_mat, preserve_order_list_sets, list_sets_highlight, list_metrics_highlight, col_name_set, list_metrics, name_set_domain_across_all = \
            summarizer._specify_summary_parameters(eval_mode=eval_mode,
                                                   is_llm=is_llm,
                                                   target_domain_mode=target_domain_mode)

        if eval_mode == "gda":
            summarizer.append_sum_confusion_matrix(["^target_(\\d|_|-)+_" + target_domain_mode + "$"],
                                                   name_set_across_domain=name_set_domain_across_all,
                                                   overwrite_input=True)
            # should reinitialize the summarizer to get the updated confusion matrix
            summarizer = ResultSummarizer(dataset, list_versions, task, model_type, eval_mode, use_post_processed_predictions=use_post_processed_predictions)
            summarizer.append_average_metric(list_metrics,
                                             list_domains_regex=["^target_(\\d|_|-)+_" + target_domain_mode + "$"],
                                             name_set_domain_across_all=name_set_domain_across_all,
                                             list_average_domain_methods=["macro", "micro"],
                                             overwrite_input=True)
        # don't visulize the metrics one by one to save space
        # - already visulized in the combined plot below
        # summarizer.visualize_metrics_over_domains_sep_metric(list_metrics, list_order_x_axis)

        summarizer.visualize_metrics_over_domains_comb_metrics(list_metrics, list_order_x_axis)
        summarizer.visualize_confusion_metrices_over_domains_comb(list_sets_confusion_mat, preserve_order_list_sets=preserve_order_list_sets)

        # should reinitialize the summarizer to get the updated metrics
        summarizer = ResultSummarizer(dataset, list_versions, task, model_type, eval_mode, use_post_processed_predictions=use_post_processed_predictions)
        # append highlight metrics to a summary csv file
        # - for gda, Q2, [target_all_macro]: f1_macro/ f1_NONE/ f1_FAVOR/ f1_AGAINST
        summarizer.write_hightlight_metrics_to_summary_csv(
            list_metrics_highlight=list_metrics_highlight,
            list_sets_highlight=list_sets_highlight,
            col_name_set=col_name_set)

    # ----------
    # comparing across model types
    # ----------
    elif isinstance(model_type, list):
        assert len(model_type) > 1
        dict_result_summarizer = dict()
        for model_type_this in model_type:
            summarizer_gda_baseline = ResultSummarizer(dataset, dict_versions[model_type_this], task,
                                                       model_type_this, eval_mode)
            dict_result_summarizer[model_type_this] = summarizer_gda_baseline

        if len(list_gpt_model_types) > 0:
            for gpt_model_type_this in list_gpt_model_types:
                summarizer_gpt = ResultSummarizer(
                    dataset, dict_versions["llm_" + gpt_model_type_this],
                    task, "llm_" + gpt_model_type_this,
                    eval_mode)
                dict_result_summarizer["llm_" + gpt_model_type_this] = summarizer_gpt

        assert len(dict_result_summarizer) == len(model_type)
        result_comparer = MultiModelTypeComparer(dataset, task, dict_result_summarizer, model_type, eval_mode)
        # get the parameters for summarizing the results
        list_order_x_axis, list_sets_confusion_mat, preserve_order_list_sets, col_name_set, list_metrics, name_set_domain_across_all = \
            result_comparer._specify_summary_parameters(eval_mode=eval_mode, target_domain_mode=target_domain_mode)

        # visulize the metrics across model types
        result_comparer.visualize_compare_models_across_domains(list_metrics, list_order_x_axis)

        # write highlight metrics to a summary csv file
        result_comparer.write_hightlight_metrics_to_summary_csv(
            col_name_set=col_name_set)
