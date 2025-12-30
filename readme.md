
## Overview

This project consists of two main stages:

1. **Fine-tuning (Finetune)**  
2. **Feature Extraction and Downstream Task Evaluation**

All configurations are managed through YAML files under the `cfgs/` directory, and datasets are organized in a subject-wise manner.

---

## Pipeline

### Step 1: Fine-tuning

Run the fine-tuning script:

```bash
python finetune.py
````

* This script loads configuration from:

  ```text
  cfgs/arg_ft.yaml
  ```
* The dataset used for fine-tuning is specified in the YAML file via:

  ```yaml
  DATA_ROOT_DIR/DATASET_NAME
  ```

---

### Step 2: Feature Extraction and Downstream Evaluation

Run the feature extraction and downstream task evaluation script:

```bash
python ext_down.py
```

* This script loads configuration from:

  ```text
  cfgs/arg_ex.yaml
  ```
* The dataset path is also specified via:

  ```yaml
  DATA_ROOT_DIR/DATASET_NAME
  ```

---

## Dataset Organization

The dataset directory specified by `DATA_ROOT_DIR/DATASET_NAME` should follow the structure below:

```text
DATASET_NAME/
├── sub_1/
│   ├── data.mat
│   └── label.mat
├── sub_2/
│   ├── data.mat
│   └── label.mat
├── ...
└── sub_n/
    ├── data.mat
    └── label.mat
```

* The dataset contains `n_sub` folders, each corresponding to one subject.
* Subject folders are named as:

  ```text
  sub_1, sub_2, ..., sub_n
  ```

---

## Data Format Details

### `data.mat`

Each `data.mat` file contains **two objects**:

1. **data**

   * Shape:

     ```text
     [n_channel, n_sample_point]
     ```
   * Description:

     * Multidimensional time-series signal.
     * `n_sample_point = trial_duration_in_seconds × sampling_rate`.

2. **trial_duration**

   * Shape:

     ```text
     [1, n_trial]
     ```
   * Description:

     * Each element is a `float` representing the duration (in seconds) of a trial.
     * There are `n_trial` trials in total.

---

### `label.mat`

* Contains `n_trial` label matrices (trial_{trial_id}).
* Each matrix corresponds to the label of one trial for the subject. T of can be of arbitrary length (it will be resampled to match the trial length), representing the multidimensional label dynamics of the subject for that trial. If no dynamic in trial (same label for all samples from one trial, like emotion class for vedio), T can be 1.

**trial_{trial_id}:**

* Shape:

  ```text
  [n_dim, T]
  ```

  where:

  * `n_dim`: label dimension
  * `T`: time dimension (may vary across trials)

---

