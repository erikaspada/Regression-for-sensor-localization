import pandas as pd
import numpy as np


def load_data(dev_path: str, eval_path: str):
    df_dev = pd.read_csv(dev_path)
    df_eval = pd.read_csv(eval_path)
    return df_dev, df_eval


def remove_noisy_features(df):
    noisy_indices = [0, 7, 12, 15, 16, 17]
    columns_to_remove = [
        f"tmax[{i}]" for i in noisy_indices
    ] + [
        f"pmax[{i}]" for i in noisy_indices
    ] + [
        f"negpmax[{i}]" for i in noisy_indices
    ] + [
        f"rms[{i}]" for i in noisy_indices
    ] + [
        f"area[{i}]" for i in noisy_indices
    ]
    return df.drop(columns=columns_to_remove)


def remove_incorrect_data(df):
    valid_indices = [1,2,3,4,5,6,8,9,10,11,13,14]
    for i in valid_indices:
        df = df[df[f'negpmax[{i}]'] <= 0]
        df = df[df[f'pmax[{i}]'] >= 0]
    return df


def remove_outliers(df):
    mask = (
        (df['negpmax[1]'] < -70) |
        (df['negpmax[2]'] < -40) |
        (df['negpmax[3]'] < -80) |
        (df['negpmax[4]'] < -50) |
        (df['negpmax[5]'] < -80) |
        (df['negpmax[6]'] < -75) |
        (df['negpmax[8]'] < -75) |
        (df['negpmax[9]'] < -70) |
        (df['negpmax[10]'] < -85) |
        (df['negpmax[11]'] < -80) |
        (df['negpmax[13]'] < -80) |
        (df['negpmax[14]'] < -70)
    )
    return df[~mask]


def add_features(df):
    indices = [1,2,3,4,5,6,8,9,10,11,13,14]
    for i in indices:
        df[f'range[{i}]'] = df[f'pmax[{i}]'] - df[f'negpmax[{i}]']
    return df


def drop_low_importance_features(df):
    low_importance = [1,2,3,4,5,6,8,9,10,11,13,14]
    columns_to_remove = [f'tmax[{i}]' for i in low_importance] + [f'rms[{i}]' for i in low_importance]
    return df.drop(columns=columns_to_remove)


def preprocess_data(dev_path, eval_path):
    df_train, df_eval = load_data(dev_path, eval_path)

    df_train = remove_noisy_features(df_train)
    df_eval = remove_noisy_features(df_eval)

    df_train = remove_incorrect_data(df_train)
    df_train = remove_outliers(df_train)

    df_train = add_features(df_train)
    df_eval = add_features(df_eval)

    df_train = drop_low_importance_features(df_train)
    df_eval = drop_low_importance_features(df_eval)

    return df_train, df_eval
