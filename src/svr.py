import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
import joblib
from functools import partial
from timeit import default_timer as timer
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

'''
Implement laplacian and gaussian kernels to svr
'''


def laplacian_kernel(X, Y, gamma):
    kernel = manhattan_distances(X, Y)
    kernel = np.exp(-kernel * gamma)
    return kernel


def gaussian_kernel(X, Y, gamma):
    kernel = euclidean_distances(X, Y) ** 2
    kernel = np.exp(kernel * (-1 / (gamma ** 2)))
    return kernel


def test_svr_models(models, X, y):
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

    joblib.dump(models, '../models/svr.pkl')
    results.insert(loc=0, column='metrics', value=['RMSE', 'MAE', 'best_params'])

    return results


laplacian_params = {
    "C": [1000],
    "epsilon": [0.5]
}

gaussian_params = {
    "C": [1000],
    "epsilon": [1e-4]
}

models = {
    'Laplacian': GridSearchCV(SVR(kernel=partial(laplacian_kernel, gamma=1e-2)), cv=5,
                              param_grid=laplacian_params, n_jobs=-1),
    'Gaussian': GridSearchCV(SVR(kernel=partial(gaussian_kernel, gamma=6)), cv=5,
                             param_grid=gaussian_params, n_jobs=-1),
}

if __name__ == "__main__":
    start = timer()
    X = np.load('../input/X_minmax_scaled.npy')
    y = np.load('../input/y.npy')
    results_df = test_svr_models(models, X, y)
    end = timer()
    print("time elapsed", end - start)
    print(results_df)
    results_df.to_csv('../results/svr_results.csv', index=False)
