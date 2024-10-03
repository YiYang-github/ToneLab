
# Overview

ToneLab is an inclusive and easy-to-use platform designed for lightweight documentation and quantitative analysis in Sino-Tibetan tonal languages, which comprises 2 core modules: [Tone2Vec](#tone2vec-module), [Lightweight Documentation](#lightweight-documentation).. 

<div align="center">
<img align="center" src="docs/figures/figure1.pdf" width="1000px" />
<b><br>Figure 1.</b>
</div>

- **Tone2Vec Module**: Enables representations for phonetic analysis from tone, initial, and final transcriptions, such as 't ɔ 55'. With embeddings, you can do more large-scale quantitative studies, like language variations, evolutions, and the classification of dialects.
  
- **Automatic Transcription**: Accepts speech from any dialect as input and automatically outputs a five-scale transcription using trained ML models, such as "215" or "51".
  
- **Automatic Clustering**: Automatically determines tone categories and values from collected signals of a dialect.

Related Paper: "Automated Tone Transcription and Clustering with Tone2Vec", EMNLP 2024 Findings.

ToneLab is an early exploratory step for the revitalization of Sino-Tibetan indigenous languages by young undergrads. We hope our small effort could motivate more attention to this field. More open datasets, use cases, and potential collaborations are especially appreciated.

## More about the Proposing of ToneLab

- **The Extinction of Indigenous Languages**: Of the 6,700 languages spoken worldwide, forty percent are at risk of extinction—predominantly indigenous ones. This has become a global crisis; the United Nations General Assembly ([Resolution A/RES/74/135](https://documents.un.org/doc/undoc/gen/n19/426/26/pdf/n1942626.pdf)) proclaimed the period between 2022 and 2032 as the International Decade of Indigenous Languages (IDIL). Each language that vanishes signifies the permanent loss of unique indigenous histories, cultures, and identities.

- **Obstacles in Protection**: Current phonetic fieldwork relies on manual effort, resulting in substantial time and financial costs. This is especially challenging for the numerous endangered languages that are rapidly disappearing, often compounded by limited funding. Moreover, most NLP techniques are built on majority languages, like Mandarin and English, making lightweight documentation tools difficult to develop.

- **Obstacles in Analysis**: Several fieldworks have gathered abundant tone transcription data, represented by the *Atlas of the World's Languages in Danger* (UNESCO) and [Chinese Language Resources Protection Project](http://www.china-language.gov.cn/). This has created an urgent need to develop comparable features for different tone, initial, and final transcriptions and to use computational methods to analyze variations across these dialect regions.


## Installation

Prebuilt ToneLab can be directly installed with `pip` (tested with Python 3.8 and above): 

```bash
pip install tonelab
```



# Tone2Vec Module

<div align="center">
<img align="center" src="docs/figures/figure2.png" width="1000px" />
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



# Lightweight Documentation

ToneLab enables automatic tone transcription and clustering by training machine learning models. Currently, we support MLP and CNN models, including ResNet, VGG, and DenseNet. Users can use the provided models or train their own models with their own data. 
