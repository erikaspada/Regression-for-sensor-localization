from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor

def get_model():
    base_estimator = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        bootstrap=False,
        n_jobs=-1
    )
    return MultiOutputRegressor(base_estimator)


def train_model(df_train):
    X_train = df_train.drop(['x', 'y'], axis=1)
    y_train = df_train[['x', 'y']]
    model = get_model()
    model.fit(X_train, y_train)
    return model
