import pandas as pd
import numpy as np

class Model:
    def __init__(self):
        # Training data
        self._X_train = None
        self._y_train = None

        # Fitted solution
        self.alpha_hat = None

    def k_matrix(self, X1, X2, gamma):

        X1_sq = np.sum(X1 * X1, axis=1)[:, None]
        X2_sq = np.sum(X2 * X2, axis=1)[None, :]
        sq_dist = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)
        sq_dist = np.maximum(sq_dist, 0.0)

        return np.exp(-gamma * sq_dist)
    
    def fit(self, X_train, y_train, lam, gamma):

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.to_numpy()

        self._X_train = X_train
        self._y_train = y_train

        K = self.k_matrix(X_train, X_train, gamma)
        I = np.eye(K.shape[0])

        A = K + lam * I
        alpha_hat = np.linalg.solve(A, y_train)

        self.alpha_hat = alpha_hat

    def predict(self, X_test, X_train, gamma):

        if self.alpha_hat is None:
            raise ValueError('Model not fitted')

        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()

        K = self.k_matrix(X_test, X_train, gamma=gamma)

        y_pred = K @ self.alpha_hat
        return y_pred
    
    def save_csv(self, y_pred):
        np.savetxt('data/result.csv', y_pred, delimiter=',', header='price_CHF', comments='') 

def load_dfs(for_cv=False):
    # loads and preprosseses the trainig data
    train_df = pd.read_csv(filepath_or_buffer='data/train.csv')

    # remove all columns with missing label
    train_df = train_df.dropna(subset=['price_CHF'])

    # seperating the training data
    X_train = train_df.drop(columns=['price_CHF'])
    X_train = preprosses(X_train, for_cv)

    # No need to preprosess the y_data since all Nan are allready removed
    y_train = train_df[['price_CHF']]

    # loads and preprosseses the testing data (no price_CHF allready)
    X_test = pd.read_csv(filepath_or_buffer='data/test.csv')
    X_test = preprosses(X_test, for_cv)
    
    # checking that the dimensions is correct
    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"

    return X_train, y_train, X_test

def preprosses(df:pd.DataFrame, for_cv:bool):

    # creating dummy columns for the seasons [spring, summer, autumn, winter]
    if ('season' in df.columns):
        df = pd.get_dummies(df, columns=['season'], dtype=int)
    
    # stitching togheter the time-series for each column
    if for_cv:
        pass
    else:
        df = df.interpolate().ffill().bfill()
    
    return df

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def cross_validate(X_train, y_train, param_dict: dict, k=5):

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k)

    lambdas = param_dict.get('lambdas') # e.g [1, 0.1, 0.01, 0.001, 0.00001]
    gammas = param_dict.get('gammas')# e.g [0.1, 1, 5, 10, 100]

    n_l = len(lambdas)
    n_g = len(gammas)

    results = np.zeros(shape=(n_l, n_g))

    for i in range(n_l):
        lam = lambdas[i]
        for j in range(n_g):
            gamma = gammas[j]

            mse_list = []

            for train_idx, val_idx in kf.split(X_train):
                X_train_fold_df = X_train.iloc[train_idx].copy().ffill().bfill()
                X_val_fold_df = X_train.iloc[val_idx].copy().ffill().bfill()

                y_train_fold_df = y_train.iloc[train_idx].copy()
                y_val_fold_df = y_train.iloc[val_idx].copy()

                model = Model()
                model.fit(X_train=X_train_fold_df.to_numpy(), 
                          y_train=y_train_fold_df.to_numpy(), 
                          lam=lam, 
                          gamma=gamma
                )
                y_pred = model.predict(X_test=X_val_fold_df.to_numpy(), X_train=X_train_fold_df.to_numpy(), gamma=gamma)

                mse = mean_squared_error(y_val_fold_df.to_numpy(), y_pred)
                mse_list.append(mse)
            
            results[i, j] = np.mean(mse_list)

    return results

def visualize_tuning(param_dict, cv_result):
    import plotly.graph_objects as go
    lambdas = param_dict.get('lambdas')
    gammas = param_dict.get('gammas')

    G, L = np.meshgrid(gammas, lambdas)

    fig = go.Figure(data=[
      go.Surface(x=L, 
                 y=G, 
                 z=cv_result, 
                 cmin=0,
                 cmax=12,
                 colorscale='jet', 
                 opacity=0.7)
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title="lambda",
            yaxis_title='gamma',
            zaxis_title="CV MSE",
            xaxis_type="log",
            yaxis_type="log",
            zaxis=dict(range=[0, 12])
    ))

    fig.show()

def main():

    # Cross validation
    X_train, y_train, X_test = load_dfs(for_cv=True)

    param_dict = {
    "lambdas": np.logspace(-16, 3, 20),
    "gammas":  np.logspace(-16, 3, 20)
    }

    cv_result = cross_validate(X_train=X_train, y_train=y_train, param_dict=param_dict)
    np.set_printoptions(suppress=True)

    visualize_tuning(param_dict=param_dict, cv_result=cv_result)

    # best params: lam = 0.1, gamma = 0.15

    # Training
    X_train, y_train, X_test = load_dfs(for_cv=False)
    lam = 0.1
    gamma = 0.15

    model = Model()
    model.fit(X_train, y_train, lam=lam, gamma=gamma)
    y_pred = model.predict(X_test, X_train, gamma=gamma)

    model.save_csv(y_pred)

if __name__ == '__main__':
    main()