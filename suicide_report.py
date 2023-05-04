import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data exploration
# ============================================================ #
df = pd.read_csv("Suicides in India 2001-2012.csv")
print("Missing Instances attribute wise:\n", df.isna().sum())
print(df.describe())

print(df["State"].unique())
dfilt = df[df["State"] != "Total (All India)"]
dfilt = dfilt[dfilt["State"] != "Total (States)"]
dfilt = dfilt[dfilt["State"] != "Total (Uts)"]
print(dfilt.shape)

print(dfilt["State"].unique())
print(dfilt["Type_code"].unique())
newdf = dfilt.drop(columns="Type", axis=1)
print(newdf.sample(10, random_state=101))


plt.figure(figsize=(8, 5))
sns.barplot(x="Year", y="Total", data=df[df["State"]=="Total (All India)"], hue="Gender", errorbar=None, palette="RdPu")
plt.xlabel("Year")
plt.ylabel("Total suicides")
plt.title("Total suicides in years by gender (India 2001-2012)")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="Year", y="Total", data=newdf, hue="Gender", errorbar=None, palette="RdPu")
plt.xlabel("Year")
plt.ylabel("Total suicides")
plt.title("Total suicides in years by gender (India 2001-2012)")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="Age_group", y="Total", hue="Gender", data=newdf, errorbar=None, palette="turbo")
plt.xlabel("Age group")
plt.ylabel("Total suicides")
plt.title("Total suicides for different age groups divided by gender")
plt.show()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(7, 15))
sns.barplot(x="Total", y="State", hue="Gender", data=newdf, orient="h", errorbar=None)
plt.title("State wise total suicides divided by gender")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="Type_code", y="Total", hue="Gender", data=newdf, errorbar=None, palette="rainbow")
plt.title("Type code based total suicides divided by gender")
plt.show()

# KNN
# ============================================================ #
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

dfgrp = (
    newdf.groupby(by=["State", "Year", "Type_code", "Gender", "Age_group"])
    .sum()
    .reset_index()
)
dfgrp.sample(10, random_state=101)
pd.set_option("display.max_columns", 100)
dfmod = pd.get_dummies(dfgrp, prefix=["state", "type_code", "Gender", "Agegrp"])
dfmod.sample(10, random_state=101)
X = np.array(dfmod.drop(columns="Total", axis=1))
y = np.array(dfmod["Total"])

sel = VarianceThreshold(threshold=(0.9 * (1 - 0.9)))
Xmod = sel.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xmod, y, test_size=0.3, random_state=559)

mae = []
for k in range(1, 16):
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(X_train, y_train)
    ypreds = neigh.predict(X_test)
    mae.append(np.mean(abs(y_test - ypreds)))
plt.figure(figsize=(8, 5))
plt.grid(True)
plt.plot(range(1, 16), mae, "mo-")
plt.xlabel("k")
plt.ylabel("Mean Absolute error")
plt.title("Optimal k by using Elbow method")
plt.show()

neigh = KNeighborsRegressor(n_neighbors=10)
neigh.fit(X_train, y_train)
print("Training set R^2 score:", neigh.score(X_train, y_train))
print("K nearest neighbor regressor model parameters:\n", neigh.get_params())
ypreds = neigh.predict(X_test)

# Regression report
# ============================================================ #
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    max_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
)

def regression_report(y_true, y_pred):
    error = y_true - y_pred
    percentil = [5, 25, 50, 75, 95]
    percentil_value = np.percentile(error, percentil)
    metrics = [
        ("mean absolute error", mean_absolute_error(y_true, y_pred)),
        ("median absolute error", median_absolute_error(y_true, y_pred)),
        ("mean squared error", mean_squared_error(y_true, y_pred)),
        ("max error", max_error(y_true, y_pred)),
        ("r2 score", r2_score(y_true, y_pred)),
        ("explained variance score", explained_variance_score(y_true, y_pred)),
    ]
    print("Metrics for regression:")
    for metric_name, metric_value in metrics:
        print(f"{metric_name:>25s}: {metric_value: >20.3f}")
    print("\nPercentiles:")
    for p, pv in zip(percentil, percentil_value):
        print(f"{p: 25d}: {pv:>20.3f}")
    res = pd.Series(y_true - y_pred)
    print("Residual Statistics:\n", res.describe())
    plt.figure(figsize=(8, 5))
    plt.plot(y_true - y_pred, "r-.")
    plt.grid(True)
    plt.xlabel("Instance number in test set")
    plt.ylabel("Residual error")
    plt.title("Residual plot via KNN regressor model")
    plt.show()

regression_report(y_test, ypreds)

# Decision tree
# ============================================================ #
from sklearn.tree import DecisionTreeRegressor

treemodel = DecisionTreeRegressor(random_state=101, ccp_alpha=1e-6, max_leaf_nodes=150).fit(X_train, y_train)
print("Decision tree model parameters:\n", treemodel.get_params())
print("Training set R^2 value:", treemodel.score(X_train, y_train))
ypreds = treemodel.predict(X_test)

regression_report(y_test, ypreds)
res = pd.Series(y_test - ypreds)
print("Residual Statistics:\n", res.describe())
plt.figure(figsize=(8, 5))
plt.plot(y_test - ypreds, "b-.")
plt.grid(True)
plt.xlabel("Instance number in test set")
plt.ylabel("Residual error")
plt.title("Residual plot via Decision tree regressor model")
plt.show()

# SGD regression
# ============================================================ #
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sgdmodel = make_pipeline(
    StandardScaler(), SGDRegressor(max_iter=400, tol=1e-4, learning_rate="adaptive")
)
print("SGD model parameters:\n", sgdmodel.get_params())
sgdmodel.fit(X_train, y_train)
print("Training set R^2 value: ", sgdmodel.score(X_train, y_train))
ypreds = sgdmodel.predict(X_test)
regression_report(y_test, ypreds)

# Support Vector Regression / Machine
# ============================================================ #
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

svrmddel = make_pipeline(StandardScaler(), SVR(kernel="linear", max_iter=-1)).fit(X_train, y_train)
print("SVR model parameters:\n", svrmddel.get_params())
print("Training set R^2 value: ", svrmddel.score(X_train, y_train))
ypreds = svrmddel.predict(X_test)

# Linear regression
# ============================================================ #
from sklearn.linear_model import LinearRegression
linreg = LinearRegression().fit(X_train, y_train)
print("Train set R^2 score by linear regression model:", linreg.score(X_train, y_train))
print("Linear regression model coefficients:\n", linreg.coef_)
ypreds = linreg.predict(X_test)

# Adaboost
# ============================================================ #
from sklearn.ensemble import AdaBoostRegressor
adaboostmodel = AdaBoostRegressor( random_state=101, 
                                   n_estimators=100, 
                                   learning_rate=1e-4, 
                                   loss="exponential").fit(X_train, y_train)
print("Adaboost model parameters:\n", adaboostmodel.get_params())
print("Train set R^2 score by AdaBoost regression model:", adaboostmodel.score(X_train, y_train))
ypreds = adaboostmodel.predict(X_test)

# Random forest
# ============================================================ #
from sklearn.ensemble import RandomForestRegressor
rfmodel = RandomForestRegressor( max_depth=30,
                                 random_state=101,
                                 ccp_alpha=1e-2,
                                 n_estimators=200,
                                 max_leaf_nodes=150,
                                 max_features=6,).fit(X_train, y_train)
print("Random Forest regressor model parameters:\n", rfmodel.get_params())
print( "Train set R^2 score by Random Forest regression model:", rfmodel.score(X_train, y_train))
ypreds = rfmodel.predict(X_test)

# MLP
# ============================================================ #
from sklearn.neural_network import MLPRegressor
mlpnnet = MLPRegressor( random_state=101,
                        solver="lbfgs",
                        hidden_layer_sizes=(40, 30),
                        activation="identity",
                        max_iter=700,
                        early_stopping=True ).fit(X_train, y_train)
print("MLP neural network regressor model parameters:\n", mlpnnet.get_params())
print("Train set R^2 score by MLP neural network regression model:", mlpnnet.score(X_train, y_train))
ypreds = mlpnnet.predict(X_test)


# Missing Instances attribute wise:
#  State        0
# Year         0
# Type_code    0
# Type         0
# Gender       0
# Age_group    0
# Total        0
# dtype: int64

# Training set R^2 score: 0.16686688321450782
# K nearest neighbor regressor model parameters:
#  {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 10, 'p': 2, 'weights': 'uniform'}
# Metrics for regression:
#       mean absolute error:              613.031
#     median absolute error:              287.050
#        mean squared error:          1406502.895
#                 max error:             9924.800
#                  r2 score:                0.143
#  explained variance score:                0.144

# Percentiles:
#                         5:            -1121.790
#                        25:             -420.250
#                        50:              -73.100
#                        75:               72.700
#                        95:             1978.590
# Residual Statistics:
#  count    4284.000000
# mean       47.983427
# std      1185.127985
# min     -3068.300000
# 25%      -420.250000
# 50%       -73.100000
# 75%        72.700000
# max      9924.800000
# dtype: float64

# Decision tree model parameters:
#  {'ccp_alpha': 1e-06, 'criterion': 'squared_error', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 150, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 101, 'splitter': 'best'}
# Training set R^2 value: 0.20565711545621768
# Metrics for regression:
#       mean absolute error:              605.008
#     median absolute error:              294.318
#        mean squared error:          1313385.042
#                 max error:             9076.295
#                  r2 score:                0.200
#  explained variance score:                0.200

# Percentiles:
#                         5:            -1225.817
#                        25:             -439.869
#                        50:              -80.201
#                        75:               56.739
#                        95:             1949.937
# Residual Statistics:
#  count    4284.000000
# mean       23.548283
# std      1145.921918
# min     -2433.955556
# 25%      -439.868687
# 50%       -80.201327
# 75%        56.739130
# max      9076.295455
# dtype: float64
# Residual Statistics:
#  count    4284.000000
# mean       23.548283
# std      1145.921918
# min     -2433.955556
# 25%      -439.868687
# 50%       -80.201327
# 75%        56.739130
# max      9076.295455
# dtype: float64

# SGD model parameters:
#  {'memory': None, 'steps': [('standardscaler', StandardScaler()), ('sgdregressor', SGDRegressor(learning_rate='adaptive', max_iter=400, tol=0.0001))], 'verbose': False, 'standardscaler': StandardScaler(), 'sgdregressor': SGDRegressor(learning_rate='adaptive', max_iter=400, tol=0.0001), 'standardscaler__copy': True, 'standardscaler__with_mean': True, 'standardscaler__with_std': True, 'sgdregressor__alpha': 0.0001, 'sgdregressor__average': False, 'sgdregressor__early_stopping': False, 'sgdregressor__epsilon': 0.1, 'sgdregressor__eta0': 0.01, 'sgdregressor__fit_intercept': True, 'sgdregressor__l1_ratio': 0.15, 'sgdregressor__learning_rate': 'adaptive', 'sgdregressor__loss': 'squared_error', 'sgdregressor__max_iter': 400, 'sgdregressor__n_iter_no_change': 5, 'sgdregressor__penalty': 'l2', 'sgdregressor__power_t': 0.25, 'sgdregressor__random_state': None, 'sgdregressor__shuffle': True, 'sgdregressor__tol': 0.0001, 'sgdregressor__validation_fraction': 0.1, 'sgdregressor__verbose': 0, 'sgdregressor__warm_start': False}
# Training set R^2 value:  0.1877648481689056
# Metrics for regression:
#       mean absolute error:              611.700
#     median absolute error:              311.579
#        mean squared error:          1304914.383
#                 max error:             9465.106
#                  r2 score:                0.205
#  explained variance score:                0.205

# Percentiles:
#                         5:            -1445.430
#                        25:             -444.685
#                        50:             -153.027
#                        75:              115.020
#                        95:             1854.879
# Residual Statistics:
#  count    4284.000000
# mean       23.232132
# std      1142.225546
# min     -1820.882219
# 25%      -444.684679
# 50%      -153.027359
# 75%       115.019637
# max      9465.106000
# dtype: float64

# SVR model parameters:
#  {'memory': None, 'steps': [('standardscaler', StandardScaler()), ('svr', SVR(kernel='linear'))], 'verbose': False, 'standardscaler': StandardScaler(), 'svr': SVR(kernel='linear'), 'standardscaler__copy': True, 'standardscaler__with_mean': True, 'standardscaler__with_std': True, 'svr__C': 1.0, 'svr__cache_size': 200, 'svr__coef0': 0.0, 'svr__degree': 3, 'svr__epsilon': 0.1, 'svr__gamma': 'scale', 'svr__kernel': 'linear', 'svr__max_iter': -1, 'svr__shrinking': True, 'svr__tol': 0.001, 'svr__verbose': False}
# Training set R^2 value:  -0.06166857723527586
# Train set R^2 score by linear regression model: 0.18776486487716937

# Linear regression model coefficients:
#  [   9.02441782 -292.56100822 -281.41043542 -284.11319515 -127.9912431
#   127.9912431   858.08463878 -474.43195131   71.64519573   77.66556241
#  -155.88434534 -377.07910028]

# Adaboost model parameters:
#  {'base_estimator': 'deprecated', 'estimator': None, 'learning_rate': 0.0001, 'loss': 'exponential', 'n_estimators': 100, 'random_state': 101}
# Train set R^2 score by AdaBoost regression model: 0.183522926727811

# Random Forest regressor model parameters:
#  {'bootstrap': True, 'ccp_alpha': 0.01, 'criterion': 'squared_error', 'max_depth': 30, 'max_features': 6, 'max_leaf_nodes': 150, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 200, 'n_jobs': None, 'oob_score': False, 'random_state': 101, 'verbose': 0, 'warm_start': False}
# Train set R^2 score by Random Forest regression model: 0.2057634885227818

# MLP neural network regressor model parameters:
#  {'activation': 'identity', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': True, 'epsilon': 1e-08, 'hidden_layer_sizes': (40, 30), 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 700, 'momentum': 0.9, 'n_iter_no_change': 10, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 101, 'shuffle': True, 'solver': 'lbfgs', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}
# Train set R^2 score by MLP neural network regression model: 0.1863174619350212
# [Finished in 33.1s]
