
# Overview

ToneLab is an inclusive and easy-to-use platform designed for lightweight documentation and quantitative analysis in Sino-Tibetan tonal languages, which comprises 2 core modules: [Tone2Vec](#tone2vec), [Lightweight Documentation](#documentation). 

<div align="center">
<img align="center" src="docs/figures/figure3.png" width="1000px" />
<b><br>Figure 1.</b>
</div>

- **Tone2Vec Module**: Enables representations for phonetic analysis from tone, initial, and final transcriptions, such as 't ɔ 55'. With embeddings, you can do more large-scale quantitative studies, like language variations, evolutions, and the classification of dialects.
  
- **Automatic Transcription**: Accepts speech from any dialect as input and automatically outputs a five-scale transcription using trained ML models, such as "215" or "51".
  
- **Automatic Clustering**: Automatically determines tone categories and values from collected signals of a dialect.

Related Paper: "Automated Tone Transcription and Clustering with Tone2Vec", EMNLP 2024 Findings.

ToneLab is an early exploratory step for the revitalization of Sino-Tibetan indigenous languages by young undergrads. We hope our small effort could motivate more attention to this field. More open datasets, use cases, and potential collaborations are especially appreciated.

## More about the Proposing of ToneLab

- **The Extinction of Indigenous Languages**: Of the 6,700 languages spoken worldwide, forty percent are at risk of extinction—predominantly indigenous ones. This has become a global crisis; the United Nations General Assembly (\href{https://documents.un.org/doc/undoc/gen/n19/426/26/pdf/n1942626.pdf}{Resolution A/RES/74/135}) proclaimed the period between 2022 and 2032 as the International Decade of Indigenous Languages (IDIL). Each language that vanishes signifies the permanent loss of unique indigenous histories, cultures, and identities.

- **Obstacles in Protection**: Current phonetic fieldwork relies on manual effort, resulting in substantial time and financial costs. This is especially challenging for the numerous endangered languages that are rapidly disappearing, often compounded by limited funding. Moreover, most NLP techniques are built on majority languages, like Mandarin and English, making lightweight documentation tools difficult to develop.

- **Obstacles in Analysis**: Several fieldworks have gathered abundant tone transcription data, represented by the *Atlas of the World's Languages in Danger* (UNESCO) and \href{http://www.china-language.gov.cn/}{Chinese Language Resources Protection Project}. This has created an urgent need to develop comparable features for different tone, initial and final transcriptions and to use computational methods to analyze variations across these dialect regions.



## Installation

Prebuilt ToneLab can be directly installed with `pip` (tested with Python 3.8 and above): 

```bash
pip install tonelab
```

Alternatively, you can install the latest DBGym version from our code repository:

```bash
git clone https://github.com/YiYang-github/ToneLab.git
cd ToneLab
pip install -e . 
```



# Tone2Vec Module {#tone2vec}

<div align="center">
<img align="center" src="docs/figures/figure3.png" width="1000px" />
<b><br>Figure 2: **Left**: Visual simulations using transcription sequences `l₁ = (55)` (green linear curve), `l₂ = (41)` (red linear curve), and `l₃ = (312)` (blue quadratic curve). Grey shading denotes the area between `(41)` and `(312)`. **Right**: The number 2.27 with grey shading represents the calculated distance between `(41)` and `(312)`.</b>
</div>


## 0. Tone Transcription

### 0.1 Transcription System: Five-scale Marking System

We use the Five-scale Marking System, developed by Yuen-Ren Chao, which is the most widely used method for transcribing tones in the Sino-Tibetan language family. In this system, the pitch of a person's speech is divided into five relative levels: `(1)`, `(2)`, `(3)`, `(4)`, and `(5)`, where `(1)` indicates the lowest pitch and `(5)` the highest. Tones are then transcribed using sequences of two or three numbers to represent the pitch contour over time. For example, a tone that starts at the mid-level pitch and rises to the high level might be transcribed as `(35)`. The relative changes between these numbers indicate the pitch movement. For example, the tones `(53)` and `(42)` both represent a falling pitch, but the first starts at the highest level `(5)` and ends at a mid-level `(3)`, while the second starts one level lower, beginning at `(4)` and ending at `(2)`.

### 0.2 Input

You may have several transcriptions for various dialects, often documented through fieldwork according to a basic vocabulary. ToneLab supports input in formats such as XLSX, CSV, or List, as illustrated below. If you have Tones, initials, and finals, please separate them with spaces. You can also refer to the [folder]() for more examples.


| \textbf{Dialect} | \textbf{Word 0} | \textbf{Word 1} | \textbf{...} | \textbf{Word n} |
| :--------------- | :-------------: | :-------------: | :----------: | :-------------: |
| 0                |       15        |       215       |              |       52        |
| 1                |       55        |       15        |              |       51        |
| 2                |       25        |       214       |              |       53        |
| 3                |       14        |       312       |              |  \textbf{N/A}   |



| \textbf{Dialect} | \textbf{Word 0} | \textbf{Word 1} | \textbf{...} | \textbf{Word n} |
| :--------------- | :-------------: | :-------------: | :----------: | :-------------: |
| 0                |     t ɔ 55      |     th ɔ 55     |              |     t ai 31     |
| 1                |     t o 45      |     th o 45     |              |     t a 213     |
| 2                |     t o 55      |     th o 55     |              |     t ai 21     |
| 3                |     t ɔ 55      |     th ɔ 55     |              |     t ai 21     |

### 1. Usgae

After loading the data, you can get representations for dialects. Then, you can do quantitaive studies more easily. For example, you can visualize dialects with tonal features.


```Python
from tonelab.tone2vec import loading, parse_phonemes, tone_feats, plot

dataset_path, dataset_info = 'tests/examples/dataset.csv', 'tests/examples/info.csv' 
dataset, labels = loading(dataset_path), loading(dataset_info, column_name='areas')
initial_list, final_list, all_list,  tone_list = parse_phonemes(dataset)
feats = tone_feats(tone_list)
plot(feats, labels)
```

<div align="center">
<img align="center" src="docs/figures/figure3.png" width="1000px" />
<b><br>Figure 3: **Left**: Automatic clustering results using DBSCAN on different dialects constructed based on Levenshtein Distance. **Right**: Label Categories of language areas in the dataset.</b>
</div>




# Documentation Module {#documentation}

ToneLab enables automatic tone transcription and clustering by training machine learning models. Currently, we support MLP and CNN models, including ResNet, VGG, and DenseNet. Users can use the provided models or train their own models with their own data. 










































# DBGym makes ML on Relational DBs simple

In the current release, DBGym focuses on relational databases (databases with multiple tables), which can be as simple as a directory of CSV or Parquet files. Here we show a simple example of what DBGym can help you achieve.

## Example DBGym use case

You may have the following CSV files in your directory `database`, each representing a table
```bash
 database
 ├── user.csv
 ├── item.csv
 └── trans.csv
```

For example, `user.csv` may look like this:

| _user | x   | y       | time       |
| ----- | --- | ------- | ---------- |
| 0     | 0.1 | A       | 2021/01/01 |
| 1     | 0.8 | A       | 2022/02/01 |
| 2     | 0.5 | B       | 2023/05/01 |
| 3     | 0.5 | **N/A** | 2024/08/01 |

Your goal is to **leverage the entire database** (not just the `user.csv` table) to predict the unknown **N/A** values in column `y` in `user.csv`.

Let's pause here - the simple capability of predicting unknown column values is in fact extremely expressive. For example, 
- It can be used for classification or regression tasks
- It can be used for time series prediction tasks suppose the N/A values are associated with future time column values
- It can be used for recommendation tasks by predicting with respect to pairs of user and item IDs
- It can unify node, edge, and graph-level prediction tasks on graphs by representing (heterogeneous) graphs as multiple tables
- ... 

This is a non-exhaustive list. By being creative, we can do so many things with the capability of predicting unknown column values.

## What DBGym can do for you

DBGym defines a simple declarative **predictive query** to describe the prediction task above, by specifying `table_name.column_name` in DBGym config. In the example above, the predictive query should be `user.y`, indicating that you want to set up an ML prediction task for column `y` in table `user`, based on the full database's information. DBGym will then automatically set up the ML prediction pipeline, including splitting the train/val/test sets, training an ML model, getting evaluation metrics for train/val/test sets, and returning `user_pred.csv` where any missing N/A values will be replaced with DBGym predictions.

To solve a given predictive query, DBGym provides a variety of ML models from tabular ML to graph ML communities, including: 

| DBGym data API      | Model                    |
| ------------------- | ------------------------ |
| Tabular             | XGBoost, MLP             |
| Graph               | GCN, GraphSAGE, GIN, GAT |
| Heterogeneous graph | HGCN, HGT                |

You can also easily create your own models and register them to DBGym. We hope to integrate more ML models into DBGym, and we sincerely appreciate your pull requests!


# DBGym Quick Start

## 0 Your first DBGym experiment

With less than 5 lines of code, you can set up and carry out experiments using DBGym.

```Python
from dbgym.run import run
from dbgym.config import get_config

# Get default DBGym config
config = get_config()
stats = run(config)
```

The experiment statistics and predictions are returned from the `run()` function and are further saved in the `output` directory by default.

## 1 Run DBGym experiments on RDBench datasets

For users who want to customize DBGym experiments, we provide two ways to customize configuration: using Python and/or using the `YAML` config file. Please refer to [`dbgym/config.py`](dbgym/config.py) for all the available configurations.

**Customize configs with Python**

You can easily set customized config values in the Python code, e.g., picking an RDBench dataset and writing your favorite predictive query. For example:

```Python
from dbgym.run import run
from dbgym.config import get_config

config = get_config()
# point to an RDBench dataset. Will auto-download if not cached
config.dataset.name = 'rdb1-ather'
# predictive query. Format: table_name.column_name
config.dataset.query = 'entry_examination.Cholesterol'
stats = run(config)
```

**Customize configs with YAML file**

Alternatively, you can write a customized config in a YAML file, e.g., `config_gcn.yaml`, and load it with DBGym:

```Python
from dbgym.run import run
from dbgym.config import get_config

config = get_config()
config.merge_from_file('config_gcn.yaml')
stats = run(config)
```

where the yaml `config_gcn.yaml` can be defined as follows:

```yaml
# Example Experiment Configuration for GCN

train:
  epoch: 200

model:
  name: 'GCN'
  hidden_dim: 128
  layer: 3

optim:
  lr: 0.001
```

## 2 Apply DBGym to your own database datasets

You can also easily apply DBGym to your own database datasets. To start, you can first organize your customized data as follows:

```
dataset_path
├── dataset_1
│   ├── user.csv
│   ├── item.csv
│   └── ...
├── dataset_2
└── ...
```

Within each CSV file, DBGym assumes that most of the table columns can be arbitrarily named, except for the key columns, where you should *follow the naming convention*, where the primary key column for table `table` is named as `_table`. For example, the column names for each CSV file may look like:
```bash
user.csv:	_user, x1, x2, ...
item.csv:	_item, Feature1, Feature2, ...
```
where the special columns begin with `_` indicate key columns, and the remaining columns (can be arbitrarily named) are regarded as feature columns. For example, `_user` is the primary key column for `user.csv`, which should save the unique ID for each user; other tables could also refer to user ID information, e.g., by adding `_user` column in `trans.csv`, in which case `_user` is the foreign key column in `trans.csv` that could have non-unique values.


Then, you can point DBGym to the dataset directory, specify the dataset you want, and write a simple predictive query by pointing to any feature column:

```Python
from dbgym.run import run
from dbgym.config import get_config

config = get_config()

# provide path to your dataset
config.dataset.dir = 'dataset_path'
config.dataset.name = 'dataset_1'
config.dataset.query = 'target.x1'
# (optional) set additional customized configs
config.merge_from_file('config_gcn.yaml')
stats = run(config)
```

Finally, DBGym will generate experiment logs and predictions in the `output` directory in `dataset_path` folder by default. The predictions for column `x1` of `target.csv` will be saved as `target_pred.csv`.

Alternatively, you can customize your output directory name by adding the following line

```Python
config.log_dir = 'output_path'
```



## 3 Include customized models to DBGym

You can further easily register your customized models to DBGym and run experiments with them. This is especially helpful if you want to benchmark your proposed model with DBGym. 
You may follow the instructions below to customize your own model and potentially contribute back to the DBGym repository via pull requests.

To start, instead of using the simple `pip install dbgym` method, you should:
```bash
git clone https://github.com/JiaxuanYou/DBGym.git
pip uninstall dbgym  # uninstall existing DBGym
pip install -e .  # install with developer mode
```

Then, you can customize your own model in `dbgym/contribute/your_model.py`:

```Python
from dbgym.register import register

class YourModel(nn.Module):
    ...

your_model_type = 'tabular_model' or 'graph_model'
register(your_model_type, 'your_model_name', YourModel)
```

Next, you can use DBGym with your customized model

```Python
from dbgym.run import run
from dbgym.config import get_config

config = get_config()
config.model.name = 'your_model_name'
stats = run(config)
```

# DBGym Feature Highlights

<div align="center">
<img align="center" src="docs/Overview.png" width="1000px" />
<b><br>Figure 1: An overview of DBGym design principle.</b>
</div>


**1. Unified Task Definition for Various Data Formats.**
- To meet the requirements of diverse users, DBGym provides 3 kinds of APIs: tabular data, homogeneous graphs, and heterogeneous graphs.
- For all these data formats, we propose a unified task definition, enabling results comparison between models for different formats of data.

**2. Hierarchical Datasets with Comprehensive Experiments** 
- Via RDBench, DBGym provides 11 datasets with a range of scales, domains, and relationships. 
- These datasets are categorized into three groups based on their relationship complexity.
- Extensive experiments with 10 baselines are carried out on these datasets.

**3. Easy-to-use Interfaces with Robust Results** 
- Highly modularized pipeline.
- Reproducible experiment configuration.
- Scalable experiment management.
- Flexible user customization.
- Results reported are averaged over the same dataset and same task type (classification or regression).

DBGym has enabled RDBench, a user-friendly toolkit for benchmarking ML methods on relational databases. Please refer to our paper for more details: *[RDBench: ML Benchmark for Relational Databases](https://arxiv.org/abs/2310.16837)*, arXiv, Zizhao Zhang*, Yi Yang*, Lutong Zou*, He Wen*, Tao Feng, Jiaxuan You.


# Logging

## Logs and predictions

After running DBGym, the logs and predictions are saved to an output directory named as `output` by default. Moreover, the command line output will also indicate the path where logs and predictions are saved. For example

```
Logs and predictions are saved to Datasets/output/rdb2-bank/loan.Status_MLP_42_20231024_123456
```

## Visualization with tensorboard

To see the visualization of the training process, you can go into the output directory and run

```bash
tensorboard --logdir .
```

Then visit `localhost:6006`, or any other port you specified.

## Citation
If you find ToneLab useful or relevant to your research, please kindly cite our paper:

```bibtex
@article{zhang2023rdbench,
  title={RDBench: ML Benchmark for Relational Databases}, 
  author={Zizhao, Zhang and Yi, Yang and Lutong, Zou and He, Wen and Tao, Feng and Jiaxuan, You},
  journal={arXiv preprint arXiv:2310.16837},
  year={2023}
}
```
