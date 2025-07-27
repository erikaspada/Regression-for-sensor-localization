import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def spatial_distribution(df):
    df_grouped = df.groupby(['x', 'y']).mean().reset_index()
    plt.figure(figsize=(13, 10))
    plt.scatter(df_grouped['x'], df_grouped['y'])
    plt.title('Spatial Distribution of Coordinates')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def pad_correlation(df, pad_index):
    features = [f'pmax[{pad_index}]', f'negpmax[{pad_index}]', f'area[{pad_index}]', f'tmax[{pad_index}]', f'rms[{pad_index}]']
    corr = df[features].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Matrix for Pad {pad_index}')
    plt.show()

def pad_boxplot(df, pad_index, feature='tmax'):
    col = f'{feature}[{pad_index}]'
    plt.figure(figsize=(7, 6))
    sns.boxplot(data=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

def spatial_feature_plot(df, feature):
    df_grouped = df.groupby(['x', 'y']).mean().reset_index()
    plt.figure(figsize=(13, 10))
    scatter = plt.scatter(df_grouped['x'], df_grouped['y'], c=df_grouped[feature], cmap='viridis')
    plt.colorbar(scatter, label=feature)
    plt.title(f'Spatial View by {feature}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def dataset_balance_plot(df):
    count_per_position = df.groupby(['x', 'y']).size().reset_index(name='count')
    plt.figure(figsize=(13, 10))
    plt.scatter(count_per_position['x'], count_per_position['y'], c=count_per_position['count'], cmap='coolwarm')
    plt.colorbar(label='Sample Count')
    plt.title('Sample Balance per Position')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
