# prelim_stance_detection


## Cobal link

- bert-tuning_colab
  - https://colab.research.google.com/drive/1nxziaKStwRnSyOLI6pLNBaAnB_aB6IsE?usp=sharing
- llm-prompting_colab.ipynb
  - https://colab.research.google.com/drive/1IFr6Iz1YH9XBWUKcWZyTU-1QtxgYqrmX?usp=sharing

## Environment Set Up


- GPU
  - Driver Version: 470.182.03
  - CUDA Version: 11.4
- conda
  - with GPU
    - `prelim`
      - `conda create -n prelim` (Python 3.10.6)
      - `conda activate prelim`
      - `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`
      - `conda install -c conda-forge accelerate`
      - `conda install autopep8`
      - `conda install -c huggingface transformers`
      - `conda install -c huggingface -c conda-forge datasets`
      - `conda install pandas scikit-learn seaborn gpustat`
      - `conda install -c conda-forge openai emoji`
      - `pip install tiktoken ipykernel`
      - `pip install sentencepiece`