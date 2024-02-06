# Adia Lab Market Prediction Competition
## A cross-section forecast problem

One of the most interesting alternative prediction problems is the problem of identifying the relative ordering in performance of an investment vehicle, in the cross-section of a pool. The goal of this competition is to rank the performance of all assets in the universe from best to worst at each given date. The target to predict in this competition is the ranking of the future performance of each asset, remapped to the interval [-1,1], and the scoring function is Spearman's rank correlation between the predicted vs true rankings.

## Dataset 
The dataset presented to the competitors is an obfuscated version of high-quality market data. Therefore, details such as the nature of each investment vehicle, the constant frequency at which dates are measured, and the definition of each feature, are not available.

### X_train:

date: A sequentially increasing integer representing a date. Time between subsequent dates is a constant, denoting an unknown but fixed frequency at which the data is sampled. The initial training dataset is composed of 268 dates. 

id: A unique identifier representing the investment vehicle at a given date. Note that the same asset has a different id at each date.

0,...,460: Anonymized features describing an investment vehicle at a given date. Derived from high-quality market data.

### y_train:

date: Same as in X_train.

id: Same as in X_train.

y: The target value to predict. It is related to the future performance of the investment vehicle at the given date. The value is normalized between -1 and 1.

### X_test:

Same structure as X_train but comprises only a few dates. This file is used to simulate the submission process locally via crunch.test().

## Statistics of the dataset

### Size of the Universe 

The pool of investment vehicles are obtained through some rule (for example S&P 500 tracks the stock performance of the 500 largest companies in the US) at different date.
Hence, the universe is evolving over time, the number of assets at time t can differ from t+1, resulting in stocks entering or existing the universe.

|       | Training |
|-------|----------|
| count | 268      |
| mean  | 2761     |
| std   | 993      |
| min   | 782      |
| max   | 4173     |

## Proposed Methodology 

To solve this task a context-aware neural network model that learns item scores by applying a self-attention mechanism has been
developed following the paper "Context-Aware Learning to Rank with Self-Attention" written by Przemysław Pobrotyn et al.'s.

### Model

Stock's target for each date are treated as tokens and stock's features as input token embeddings. We denote the length of the universe in t as 𝑙 and the
number of features as 𝑑𝑓
. Each item is first passed through a shared
fully connected layer of size 𝑑h . Next, hidden representations are
passed through an encoder part of Transformer architecture with
𝑁 encoder blocks, 𝐻 heads and hidden dimension 𝑑ℎ. An
encoder block in the Transformer consists of a multi-head attention
layer with a skip-connection to the input, followed by layer
normalisation, time-distributed feed-forward layer, and another
skip connection followed by layer normalisation. Dropout is applied
before performing summation in residual blocks. Finally, after 𝑁
encoder blocks, a fully-connected layer shared across all items in the
list is used to compute a score for each item. The model can be seen
as an encoder part of the Transformer with extra linear projection
on the input.
Since the inputs do not possess any inherent sequence, no positional encoding was incorporated into the design.

### Output Transformation
The predictions are scaled for each date to the interval [−1, 1] using a min-max scaler, leading to a constant interval across time.

### Environment Setup
To reduce the effect of extreme outliers, all features (which are continuous) are
winsorized in a cross-sectional way. A z-score threshold (-3 and 3) is used, hence,
values with z-scores beyond these thresholds are adjusted to the maximum or
minimum values among the non-outliers. The z-scores of each predictor are fitted
by using StandardScaler() on the training environment and are then used to transform the testing environment. The
function thus ”clips” extreme values to within a predetermined range, preserving
the overall data structure and reducing the influence of outliers.


The features are scaled in a cross-section way within
the range [-1, 1] by applying a MinMaxScaler() for each of them. Scaling is a
necessary step as it un-biased learning by having inputs and outputs on the
same order of magnitude and ensures that each feature contributes equally to
the decision-making process. Once again, the transformation is fitted on the
training set only, in order to ensure that no data-leakage happens, for the testing
environment the .transform() method is applied.


### Loss

The objective function to be optimized is the Spearman rank correlation loss with L1
regularization.

The specific Spearman rank correlation being not differentiable, as it operates on ranks rather than the raw data, an approximation has been desgined, by incorporating both Pearson's correlation and a penalty based on the squared differences between predicted and true values.
The use of these two components in the given formula helps the model to simultaneously learn relationships and accuracy.

L1 regularization is particularly useful for high-dimensional data, as it can encourage weight sparsity, enabling efficient feature selection by reducing the weights of certain features to zero.


### Optimization
The model is trained with the Adam optimizer for each application, leading to the minimization of the loss . Backpropagation
is performed for a maximum of 10 epochs, during which, for a given training set,
there is a train-validation split with the previous 90% of data for training and the
most recent 10% reserved for validation. Early stopping is used to prevent model
overfitting, this is triggered when the model’s spearman loss on the validation
set did not improve for 3 consecutive epochs. To prevent leakage, which takes
place when the training set contains information that also appears in the validation, the training-validation sets are separated by a gap of 1 date as the labels
are derived from 1 overlapping datapoint. Due to the stochastic nature of neural networks, the seed has been set to the default value 42 for reproducibility
and consistent results. Gradient clipping was set to the default value 1 to prevent exploding gradients during training.
The size of the universe being non-constant through time, it has been a necessary step to fix it during the model's training. There are a number of advantages to this, such as memory and computation efficiency, consistent input size, etc.
The training size has been set at the 95th percentile (3951), using padding for the shorter cases and random sampling with the longer cases. This approach ensures that the majority of stocks are adequately represented in the listings, while also dealing with long tail cases.


### Hyper-Optimization

To ensure that the model generalizes well to unseen data, a TS-CV has been implemented to respect the inherent temporal order
of the data. The training set is split
into 10 folds. It allows for multiple train-validation splits, where the validation set
is always ahead of the training split. Each training set length remains constant,
stabilizing the loss across folds, with a gap of 1 date between the training and
validation sets. By fitting and inferring on different time periods, we get a better idea of how the different hyperparameter values behave over different time periods, enabling us to select values
that generalize well. Hyperparameter optimization has been performed with 20 iterations of bayesian search, using the next search range:

parameters = {

            "attention_heads": {
                'values': [1, 4]
            },

            "dropout": {
                'max': 0.3,
                'min': 0.1
            },

            "hidden_dim": {
                'values': [16, 64, 128]
            },

            "lr": {
                'max': 0.01,
                'min': 0.0001
            },

            "n_hidden": {
                'values': [2, 4]
            },

            "batch_size": {
                'values': [2, 4, 6]
            },

            "l1reg": {
                'max': 1e-2,
                'min': 1e-4
            }
        }

To select the best model through the search iteration, a k-fold score of each candidate model was calculated as the mean of the best validation metrics minus the standard deviation of the best validation metrics. Consequently, the best model is the one with the highest k-fold score, meaning that its performances are stable across the various time-based validation splits.

## Results
This competition is focused on forecasting and has two phases. The first is the submission phase where participants can submit and test their models. The second phase, which is automatic, involves running the models against unobserved live market data.

Public Leaderboard: 216/4477 (top 4.9%)

Private Leaderboard: 250/4477 (top 5.6%)
