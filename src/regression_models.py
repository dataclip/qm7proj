import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
import joblib


def test_base_models(models, X, y):
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=234)

    for i in models:
        mse = [mean_squared_error(y[test_idx], models[i].fit(X[train_idx], y[train_idx]).predict(X[test_idx]))
               for train_idx, test_idx in kf.split(X, y)]
        rmse = [np.sqrt(mse)]

        mae = [mean_absolute_error(y[test_idx], models[i].fit(X[train_idx], y[train_idx]).predict(X[test_idx]))
               for train_idx, test_idx in kf.split(X, y)]

        results[i] = [np.mean(rmse), np.mean(mae), models[i].best_params_]
        results = pd.DataFrame(results)

    joblib.dump(models, '../models/regression.pkl')
    results.insert(loc=0, column='metrics', value=['RMSE', 'MAE', 'best_params'])

    return results


if __name__ == "__main__":

    X = np.load('../input/X_minmax_scaled.npy')
    y = np.load('../input/y.npy')

    lasso_params = {'alpha': [1e-3, 1e-1, 1, 5]}
    ridge_params = {'alpha': [1e-6, 1e-5, 1e-4, 1e-2, 1e-1]}

    models = {
                # 'OLS': linear_model.LinearRegression(),
                'Lasso': GridSearchCV(linear_model.Lasso(max_iter=5000), param_grid=lasso_params),
                'Ridge': GridSearchCV(linear_model.Ridge(max_iter=5000), param_grid=ridge_params)
            }

    results_df = test_base_models(models, X, y)
    results_df.to_csv('results/regression_results.csv', index=False)
    print(results_df)

        

