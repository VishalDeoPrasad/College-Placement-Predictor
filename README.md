## Data Visualization
Histogram:
- A histogram is a graphical representation of the distribution of a dataset. It is a way to visualize the underlying frequency distribution of a continuous variable. 
- In a histogram, the data is divided into intervals (bins), and the number of data points falling into each interval is represented by the height of a bar.

Advantage:
- Probability Distribution
- Campair 
- Most Frequent 
- Summerize

```python
#Let's plot the distribution of the data 
plt.figure(figsize=(10,6))
sns.histplot(df.cgpa,kde = True, bins=3, binwidth=0.5)
plt.title('CGPA Distribution')
```

### Check if the **CGPA** follow normal distribution
> If the p-value is greater than the significance level __(commonly 0.05)__, you would fail to reject the null hypothesis. This would suggest that there isn't enough evidence to conclude that the __CGPA__ data significantly deviates from a normal distribution.

```python
from scipy.stats import shapiro
statistic, p_value = shapiro(df.cgpa)
print(statistic,p_value)
0.9849948287010193 0.3173207938671112
```

### Scatter Plot:
- Scatter plot show the corelation between data points or big data, it show changes of one data how it effect the other data.
- it can show the outliers and we can remove it.
- Also known Scatter Plot, X-Y plot, Scatter Chart, Correllation chart.
- Help to find the corellation between 2 points, for example; 
    * study_time vs marks: Higher the study time higher the marks
    * it show the linear corellation 
- Degree of Corellation:
    * None, Low, High, Perfect
- Type of Corellation:
    * Positive: Lower to Higher
    * Negative: Higher to lower
    * Curved  : Combination of both +ve and -ve corelation
    * Partical: at a certain data it show corelation 

```python
import matplotlib.pyplot as plt
scatter = plt.scatter(df['cgpa'], df['iq'], c=df['placement'])

# Add legend based on unique values in the 'placement' column
plt.legend(*scatter.legend_elements(), title='Placement')

# Labeling the axes
plt.xlabel('CGPA')
plt.ylabel('IQ')

# Show the plot
plt.show()
```

## Optimization Techniques
>these 1, 2 optimzation techiniques are perform well when we have smaller number of parameter to be optimize. and when parameters get increase these optimization gets computational expensive. it is like an iteration where it check for one by one.
1. GridSearchCV Optimzation
1. RandomSearchCV Optimzation
1. Bayesian Optimzaiton 
>To Implement Bayesian Optimzation we are using Optuna Library.

### Randome Forest
```python
import optuna

import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def objective(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    n_estimators = trial.suggest_int("n_estimators", 100,500)

    rf = sklearn.ensemble.RandomForestClassifier(criterion =criterion,
            max_depth=max_depth, 
            n_estimators=n_estimators
        )

    score = cross_val_score(rf, X_train, y_train, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

trial = study.best_trial
print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
```

### Support Vector Machine(SVC)
```python
# Get the best parameters
best_params = study.best_params
print("Best Parameters:", best_params)
​
# Train the final model with the best parameters on the entire training set
best_svc = SVC(**best_params)
best_svc.fit(X_train, y_train)
​
# Evaluate the final model on the test set
test_accuracy = best_svc.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

Best Parameters: {'C': 32.30446637537601, 'kernel': 'rbf', 'gamma': 'scale'}
Test Accuracy: 0.75
```

## Pipeline 
>the data is first standardized using StandardScaler, and then a Support Vector Machine (SVM) classifier is applied. The entire process is encapsulated within the Pipeline.
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Define a pipeline with a preprocessing step (StandardScaler) and a classifier (SVC)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the features
    ('svm', SVC(**best_params))  # Step 2: Support Vector Machine classifier
])

# You can now use the pipeline like any other scikit-learn estimator
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(accuracy_score(pipeline.predict(X_train), y_train))
```
