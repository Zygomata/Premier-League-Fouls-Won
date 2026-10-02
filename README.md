# Football Foul-Won Prediction

This project uses football event data to predict whether a possession will result in a **foul won**. The goal is to frame the problem as a supervised machine learning task and compare several classification models while accounting for class imbalance.

## Project Overview

The dataset contains detailed football event data from matches. Because the raw dataset includes many columns and a large number of records, the project began with exploratory inspection and feature selection before training any models.

The main prediction target is:

- `wonfoul = 1`: A `Foul Won` event occurs later in the same possession.
- `wonfoul = 0`: No `Foul Won` event occurs later in the same possession.

The final model was an unweighted Random Forest classifier, selected because it provided the strongest overall balance of predictive performance, precision, recall, and ROC AUC.

## Workflow

### 1. Load and Inspect the Data

The project begins by loading raw football event data and inspecting its overall structure.
Key point to note was that on Github, the epl_event_data is too large of a file to upload onto the platform (exeeds 25mb), so a 1/28th size file was uploaded of the data. However, the whole dataset was used for the training of the model.

Initial exploration focused on:

- Dataset size and number of columns
- Data types for each feature
- Missing values
- Numeric versus categorical variables
- Columns that were irrelevant, overly detailed, or unusable for prediction

This step was important because event-level football data can include many columns that add noise without improving model performance.

### 2. Create the Target Variable

The target variable, `wonfoul`, was created by determining whether a possession later contained a `Foul Won` event.

Events were sorted by:

1. Match
2. Possession
3. Period
4. Minute
5. Second

After sorting, each event was evaluated based on whether a foul won occurred later in the same possession.

```python
wonfoul = 1  # A Foul Won event occurs later in the possession
wonfoul = 0  # No Foul Won event occurs later in the possession
```

Creating the target in this way transformed the event data into a supervised classification problem.

## Data Preparation

### Feature Selection

After creating the target, unnecessary columns were removed.

Columns were excluded when they were:

- Too detailed for the prediction task
- Unrelated to the likelihood of a foul being won
- Likely to introduce noise
- Difficult to use meaningfully in a machine learning model
- Missing too many values

Reducing the feature set helped make the modeling process more efficient and reduced the risk of overfitting.

### Class Imbalance

The target classes were imbalanced, meaning that `wonfoul = 0` appeared much more frequently than `wonfoul = 1`.

This mattered because accuracy can be misleading when one class dominates the dataset. For example, a model could achieve high accuracy by predicting the majority class most of the time while missing many foul-won possessions.

Because of this imbalance, model evaluation emphasized:

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC AUC

## Feature Engineering

Categorical features were converted into machine-learning-ready numeric features using one-hot encoding.

Examples of encoded categorical variables include:

- Possession team
- Play pattern
- Team
- Home team
- Away team
- Referee

```python
categorical_columns = [
    "possession_team",
    "play_pattern",
    "team",
    "home_team",
    "away_team",
    "referee"
]
```

One-hot encoding ensures that categorical values can be used by models such as logistic regression, decision trees, and random forests.

The training and test datasets were also aligned after encoding to ensure both datasets contained the same feature columns.

## Train-Test Split

The data was split into training and testing sets using stratification.

Stratification was used to preserve the distribution of `wonfoul` classes in both sets. This created a more realistic evaluation setup and prevented the minority class from being underrepresented in either the training or test data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

## Models Tested

Several classification models were tested to compare linear, nonlinear, and imbalance-aware approaches.

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline model for measuring initial performance |
| Balanced Logistic Regression | Logistic regression with class balancing to improve minority-class recall |
| Decision Tree | Nonlinear, rule-based model capable of capturing feature interactions |
| Random Forest | Ensemble tree model designed to reduce overfitting and improve generalization |
| Balanced Random Forest | Random forest designed to handle class imbalance more directly |

### Logistic Regression

Logistic regression was used as the baseline model.

A simpler model was useful for determining whether the relationship between the features and the target could be captured with a mostly linear decision boundary.

```python
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### Balanced Logistic Regression

Because the positive class was less common, a class-weighted logistic regression model was also tested.

```python
balanced_logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
```

The balanced version improved recall for `wonfoul = 1`, meaning it identified more possessions that eventually resulted in a foul won. However, this came with an increase in false positives.

### Decision Tree

A decision tree was tested because football event outcomes may depend on nonlinear relationships and combinations of contextual features.

For example, the likelihood of winning a foul may depend on the interaction between possession team, match state, event type, play pattern, and location.

```python
decision_tree_model = DecisionTreeClassifier(
    random_state=42
)
```

### Random Forest

A random forest classifier was used to improve on the single decision tree.

A single decision tree can overfit the training data, while a random forest combines predictions from many trees. This generally improves stability and generalization on tabular datasets.

```python
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The random forest was expected to perform well because the target depends on combinations of match context and event-level context.

### Balanced Random Forest

A balanced random forest was also tested to address the minority-class prediction challenge more directly.

```python
balanced_rf_model = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

This model improved recall for foul-won possessions but increased the number of false positives. In other words, it identified more positive cases but was less precise.

## Model Evaluation

Models were evaluated using multiple metrics rather than accuracy alone.

### Confusion Matrix

The confusion matrix was used to examine:

- True negatives
- False positives
- False negatives
- True positives

This helped show the practical tradeoff between identifying more foul-won possessions and incorrectly predicting foul-won outcomes.

### Classification Report

The classification report was used to compare:

- Precision
- Recall
- F1-score
- Support

The main focus was on performance for `wonfoul = 1`, since this was the minority class and the primary outcome of interest.

### ROC AUC

ROC AUC was used to compare how effectively each model separated positive and negative cases across different classification thresholds.

A higher ROC AUC indicates that a model is generally better at assigning higher probabilities to possessions that eventually result in a foul won.

```python
roc_auc_score(y_test, predicted_probabilities)
```

## Final Model Selection

The unweighted Random Forest was selected as the final model.

Although the balanced models improved recall for the minority class, they also increased false positives and reduced precision. The balanced random forest caught more positive cases, but it did so at the cost of overall ranking ability and prediction reliability.

The unweighted random forest provided the best overall balance between:

- Detecting foul-won possessions
- Limiting false positives
- Maintaining precision
- Achieving strong ROC AUC performance
- Generalizing well to unseen event data

## Key Takeaways

- Football event data requires substantial cleaning and feature selection before modeling.
- Creating `wonfoul` based on future events within the same possession converts raw event data into a supervised learning problem.
- Class imbalance makes accuracy an unreliable evaluation metric by itself.
- Stratified train-test splitting helps preserve class distributions during evaluation.
- One-hot encoding allows categorical match and event context variables to be used in machine learning models.
- Balanced models can improve recall for the minority class but may increase false positives.
- The unweighted Random Forest produced the best overall performance for predicting whether a possession would result in a foul won.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- Matplotlib
- Seaborn

## Future Improvements

Potential next steps for this project include:

- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- Feature importance analysis to identify the strongest predictors of foul-won possessions
- SHAP analysis for model interpretability
- Threshold tuning to optimize for precision, recall, or another business objective
- Cross-validation to evaluate model stability across multiple splits
- Additional event-location features, such as pitch coordinates and distance to goal
- Time-based or match-based splits to better simulate predictions on future matches
- Testing gradient boosting models such as XGBoost, LightGBM, or CatBoost
