# DF-SA-MRFO-SAINT-DeepFM-GF

## Bio-Inspired SAINT-DeepFM Framework for Pregnancy Outcome Prediction

### 1. Overview

This repository presents the implementation of **DF-SA-MRFO-SAINT-DeepFM-GF**, a hybrid deep learning framework for pregnancy outcome classification using heterogeneous tabular data. The proposed approach integrates Self-Attention and Intersample Attention Transformer (SAINT), Deep Factorization Machine (DeepFM), Adaptive Gated Feature Fusion (GF), and a bio-inspired Diversity-Feedback and Success-Adaptive Manta Ray Foraging Optimization (DF-SA-MRFO) algorithm.

The framework combines contextual representation learning, explicit feature-interaction modeling, adaptive feature fusion, and metaheuristic hyperparameter optimization to classify records into two categories: **Pregnant** and **Not Pregnant**.

The study evaluates the proposed framework on a publicly available benchmark containing 50,000 records, 40 input attributes, and one binary target variable.

### 2. Key Features

- **SAINT branch:** Learns contextual dependencies among heterogeneous tabular features using attention-based representation learning.
- **DeepFM branch:** Models first-order feature contributions, second-order interactions, and nonlinear higher-order interactions.
- **Adaptive Gated Feature Fusion:** Dynamically combines SAINT and DeepFM representations according to their learned contributions for each input record.
- **DF-SA-MRFO optimization:** Enhances the exploration–exploitation balance using diversity feedback, success-adaptive search, and stagnation-aware relocation.
- **Leakage-aware preprocessing:** Fits preprocessing transformations on the training subset and applies them to the validation and testing subsets.
- **Model interpretability:** Uses Kernel SHAP to investigate feature contributions to model predictions.
- **Comprehensive evaluation:** Includes baseline comparisons, component ablation, optimizer comparisons, five-fold cross-validation, and sensitivity analysis.

### 3. Proposed Methodology

The framework consists of the following stages:

1. **Dataset preparation:** Load the pregnancy outcome benchmark and identify numerical, binary, and categorical attributes.
2. **Data partitioning:** Divide the 50,000 records into training, validation, and independent testing subsets using a stratified 70:15:15 split.
3. **Preprocessing:** Perform data-quality assessment, missing-value handling, categorical encoding, and numerical feature scaling.
4. **Parallel representation learning:** Process the input through the SAINT and DeepFM branches.
5. **Adaptive gated fusion:** Project the branch representations into a common feature space and combine them using a learned gating mechanism.
6. **Outcome classification:** Pass the fused representation through the classification head to estimate the probability of the recorded pregnancy status.
7. **Hyperparameter optimization:** Apply DF-SA-MRFO using validation loss as the fitness criterion.
8. **Final model evaluation:** Train the selected configuration and evaluate the frozen model on the independent testing subset.
9. **Explainability analysis:** Apply Kernel SHAP to examine feature-level contributions.

The optimization algorithm is used during model development and is not required during inference.

### 4. Dataset Description

The benchmark contains:

- **Total records:** 50,000
- **Input attributes:** 40
- **Target variable:** `Pregnancy_Outcome`
- **Task:** Binary classification
- **Class 0:** Not Pregnant
- **Class 1:** Pregnant

The input attributes cover the following domains:

- Demographic and anthropometric characteristics
- Menstrual and reproductive history
- Endocrine and hormonal measurements
- Metabolic and nutritional indicators
- Ultrasound-derived reproductive indicators
- Lifestyle and behavioral factors
- Recorded clinical conditions
- Cardiovascular and physiological measurements

The target distribution reported in the paper comprises 15,596 records in Class 0 and 34,404 records in Class 1.


### 5. Model Architecture

#### 5.1 SAINT Branch

The SAINT branch embeds numerical and categorical inputs and processes them through stacked Transformer blocks. Attention-based operations learn contextual relationships among the input attributes.

#### 5.2 DeepFM Branch

The DeepFM branch consists of three components:

- **Linear component:** Captures first-order feature contributions.
- **Factorization machine:** Learns second-order feature interactions.
- **Deep neural component:** Models nonlinear higher-order feature relationships.

The resulting representation complements the contextual features learned by SAINT.

#### 5.3 Adaptive Gated Feature Fusion

The fusion module projects the SAINT and DeepFM representations into a common feature space. A learned gating vector adaptively determines their relative contributions before residual refinement produces the fused representation.

#### 5.4 DF-SA-MRFO Optimizer

The proposed optimizer extends manta ray foraging optimization through three principal mechanisms:

- **Diversity feedback:** Adjusts search behavior according to population diversity.
- **Success-adaptive search:** Uses recent optimization success to guide candidate updates.
- **Stagnation-aware relocation:** Relocates selected candidates around the best solution when progress stagnates.

The optimizer searches for a hyperparameter configuration that minimizes validation loss.

### 6. Experimental Configuration

The following configuration and implementation environment are reported in the manuscript.

| Parameter | Value |
|---|---|
| Programming language | Python 3.10 |
| Deep learning framework | PyTorch 2.1.0 |
| Computational platform | Google Colaboratory |
| GPU | NVIDIA Tesla T4, 16 GB |
| Random seed | 42 |
| Training epochs per optimization candidate | 50 |
| Final training epochs | 50 |
| Optimization population | 10 manta rays |
| Maximum optimization iterations | 20 |
| Stagnation tolerance | 0.0001 |
| Stagnation patience | 5 iterations |
| Success-adaptation rate | 0.20 |
| Initial exploration coefficient | 1.0 |
| Initial exploitation coefficient | 0.50 |
| Initial relocation range | 0.10 of the normalized search range |
| Loss function | Class-weighted binary cross-entropy |
| Checkpoint selection | Lowest validation loss |

The search space includes learning rate, embedding dimension, number of SAINT blocks, number of attention heads, DeepFM hidden units, dropout rate, batch size, and L2 regularization coefficient.

The reported DF-SA-MRFO run involved 415 candidate model trainings, including the initial population, iterative evaluations, and an additional stagnation-triggered relocation event. The manuscript reports an optimization duration of approximately 48 hours on the NVIDIA Tesla T4 GPU.

### 7. Software Dependencies

The manuscript reports the following software packages:

- Python 3.10
- PyTorch 2.1.0
- NumPy 1.26.4
- Pandas 2.1.4
- Scikit-learn 1.3.2
- Matplotlib 3.7.5
- Seaborn 0.13.1
- SHAP 0.43.0
- XGBoost 2.0.3
- PyTorch TabNet 4.1.0
- rtdl 0.0.13

Install the dependencies using a compatible Python environment. Package compatibility should be verified before running the complete experimental pipeline.


### 8. Evaluation Metrics

The framework is evaluated using the following classification metrics:

- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- Area under the receiver operating characteristic curve (AUROC)
- Brier score

Five-fold cross-validation, component ablation, optimizer comparisons, statistical comparisons, and sensitivity analysis are also reported in the study.

### 9. Reported Results

On the independent testing subset, the manuscript reports the following results for the proposed framework.

| Metric | Reported value |
|---|---:|
| Accuracy | 98.89% |
| Precision | 99.22% |
| Recall | 99.17% |
| Specificity | 98.29% |
| F1-score | 99.20% |
| AUROC | 0.999 |
| Brier score | 0.010 |

The reported mean accuracy under five-fold cross-validation is **98.82% ± 0.05%**.



