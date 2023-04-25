"""
The parameters across all meta-datasets.
- see parameters_*.py for the parameters specific to the each dataset
@author: Yun-Shiuan Chuang
@contact: yunshiuan.chuang@gmail.com
"""
from os.path import join


class ParametersMeta():
    # par
    # - should be adjusted manually
    DATASET_META = "SEM_EVAL"

    
    # path
    PATH_ROOT = "/content/prelim_stance_detection"
    PATH_DATA = join(PATH_ROOT, "dataset")
    PATH_RESULT = join(PATH_ROOT, "results")
    # file