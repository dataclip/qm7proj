import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from timeit import default_timer as timer
import joblib


if __name__ == "__main__":

    X = np.load('../input/X_minmax_scaled.npy')
    y = np.load('../input/y.npy')

    start = timer()

    kr = GridSearchCV(KernelRidge(kernel='laplacian'), cv=5,
                      param_grid={"alpha": [1.3e-8],
                                  "gamma": [5e-8]}, n_jobs=-1)
    # see notebook for alpha ana gamma values

    kf = KFold(n_splits=5, shuffle=True, random_state=234)

    mse = [mean_squared_error(y[test_idx], kr.fit(X[train_idx], y[train_idx]).predict(X[test_idx]))
           for train_idx, test_idx in kf.split(X, y)]

    rmse = [np.sqrt(mse)]

    mae = [mean_absolute_error(y[test_idx], kr.fit(X[train_idx], y[train_idx]).predict(X[test_idx]))
           for train_idx, test_idx in kf.split(X, y)]

    end = timer()

    print("time elapsed", end - start)

    data = {'RMSE': [np.mean(rmse)],
            'MAE': [np.mean(mae)],
            'Best_params': [kr.best_params_],
            'Best_score': [kr.best_score_]
            }

    df = pd.DataFrame(data, columns=['RMSE', 'MAE', 'Best_params', 'Best_score'])
    print(df)
    df.to_csv('../results/kernel_laplacian.csv', index=False)
    joblib.dump(kr, '../models/kernel_ridge_laplacian.pkl')
