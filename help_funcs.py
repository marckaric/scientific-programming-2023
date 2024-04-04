import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, silhouette_samples

from datetime import datetime
from tqdm import tqdm


def preprocess_and_cluster_data(df, ncom=3, num_clusters=4, rs=100001):
    """
    Preprocesses and clusters given DataFrame
    Parameter:
        df: Ein Pandas DataFrame mit den Daten.
        ncom: Die Anzahl der Komponenten der PCA.
        n_cluss: Die Anzahl der Cluster.

    Returns: 
        df_temp: clustertered and preprocessed DataFrame
    """
    try:
        df_temp = df.drop(labels=['User', 'Account'], axis=1)
    except:
        df_temp = df
    cat_cols = df_temp.select_dtypes(include=['object', 'category']).columns
    num_cols = df_temp.select_dtypes(exclude=['object', 'category']).columns
    
    column_transformer = ColumnTransformer(
        [('onehot', OneHotEncoder(sparse=False), cat_cols),
         ('scaler', MinMaxScaler(), num_cols)],
         remainder='passthrough'
    )

    pipeline = Pipeline([
        ('transformer', column_transformer),
        ('pca', PCA(n_components=ncom, random_state=rs)),
        ('kmeans', KMeans(n_clusters=num_clusters, random_state=rs))
    ])
    labels = np.array(pipeline.fit_predict(df_temp))
    df_temp['Cluster'] = labels
    df_temp['Cluster'] = df_temp.Cluster.apply(lambda n: chr(ord('A') + n))
    return df_temp


def plot_silhouette(df: pd.DataFrame, cluster_label_column: str , n_clusters: int):
    """
    Creates Silhouette Plot for Clusters in given DataFrame
    Parameters:
        df: Ein Pandas DataFrame mit den Daten.
        cluster_label_column: Der Name der Spalte im DataFrame, die die Cluster-Labels enthält.
        n_clusters: Die Anzahl der Cluster.
    """

    # Extrahieren der Daten und Cluster-Labels
    X = df.drop(cluster_label_column, axis=1).values
    cluster_labels = df[cluster_label_column].values

    # Berechnen der Silhouette-Scores für jeden Cluster
    print(f'{datetime.now()} - Calculating silhouette score...')
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f'{datetime.now()} - Calculating silhouette samples...')
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # Erstellen des Silhouette-Plots
    fig, ax = plt.subplots(figsize=(10, 7))

    y_lower, y_upper = 0, 0
    yticks = []

    for i in tqdm(range(n_clusters)):
        print(f'{datetime.now()} - Plotting: Step {i} / {n_clusters}...')
        # Berechnen der Silhouette-Werte für jeden Cluster
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper += size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)

        ax.barh(range(y_lower, y_upper),
                ith_cluster_silhouette_values,
                height=1.0,
                edgecolor='none',
                color=color)

        yticks.append((y_lower + y_upper) / 2.)
        y_lower += size_cluster_i

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks(yticks)
    ax.grid(ls=':')
    ax.set_yticklabels(range(n_clusters))
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    plt.show()


def create_boxplots(df, labels=None):
    """
    Creates Boxplot for all numerical Features of given DataFrame
    Parameters:
        df: preprocessed and clustered DataFrame
    """
    df_cluster = pd.DataFrame({'Cluster': df['Cluster']})
    df_num = df.select_dtypes(exclude=['object', 'category'])

    if not labels:
        labels=df_num.columns
    n_cols = len(labels)
    nrows, ncols = int(n_cols / 4), 4
    index = np.arange(nrows * ncols).reshape(nrows, ncols)
    row_indices, col_indices = np.indices((nrows, ncols))
    indices = list(zip(row_indices.flatten(), col_indices.flatten()))

    fig, ax = plt.subplots(nrows, ncols, figsize=(20,40))

    for idx, col in zip(indices, df_num.columns):
        sns.boxplot(data=df_cluster.join(df_num[col]), x='Cluster', y=col, ax=ax[idx], 
                    showfliers=False,
                    medianprops={"marker":"x", "markeredgecolor":"red"})
        ax[idx].set_title(f'Boxplots for {col}')
    

def create_heatmaps(df, ax=None):
    df_categorical = df.select_dtypes(include=['category', 'object'])
    if ax is None:
        fig, ax = plt.subplots(len(df_categorical.columns) - 1, 1, figsize=(8, 15))
    else:
        ax=ax

    for idx, col in enumerate(df_categorical.drop('Cluster', axis=1).columns): 
        crosstab = pd.crosstab(df_categorical['Cluster'], df_categorical[col], normalize='index') # normalize 'index' damit die unterschiedlichen Clustergroessen ausgeglichen werden
        sns.heatmap(crosstab, ax=ax[idx], cmap='Blues')
        ax[idx].set_title(f'Heatmap for {col}')


def create_single_boxplot(df, label, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    else:
        ax=ax
    df_cluster = pd.DataFrame(df.Cluster)
    sns.boxplot(data=df_cluster.join(df[label]), x='Cluster', y=label, ax=ax, 
                        showfliers=False,
                        medianprops={"marker":"x", "markeredgecolor":"red"})
    ax.set_title(f'Boxplots for {label}')


def clean_data(df):
    """
    This function is not very beatiful but runs all the necessary cleaning
    for the slurm dataset in the given order and returns the cleaned DataFrame.
    """
    def convert_to_mb(x):
        if x != 'nan':
            x = x.replace('c', '')
            x = x.replace('n', '')
        if 'M' in x:
            x = x.replace('M', '')
            return pd.to_numeric(x)
        elif 'G' in x:
            x = x.replace('G', '')
            return pd.to_numeric(x) * 1024
        else:
            return x
    df_slurm = df.copy()
    cols_missing = [col for col in df_slurm.columns if df_slurm[col].isna().sum() / len(df_slurm.index) >= 0.95] 
    cols_zero = ['ConsumedEnergy', 'ConsumedEnergyRaw', 'WCKeyID']
    cols_unique_values_only = [col for col in df_slurm.columns if len(df_slurm[col].unique()) == 1 and col not in cols_zero and col not in cols_missing]
    identifier_cols = ['AssocID', 'JobID', 'JobIDRaw']
    df_slurm.drop(labels=cols_missing, axis=1, inplace=True)
    df_slurm.drop(labels=cols_zero, axis=1, inplace=True)
    df_slurm.drop(labels=cols_unique_values_only, axis=1, inplace=True)
    df_slurm.drop(labels=identifier_cols, axis=1, inplace=True) 
    df_slurm.drop(labels=['End', 'CPUTime', 'CPUTimeRAW',
                           'Elapsed', 'Timelimit', 'ResvCPU',
                           'QOS', 'Eligible', 'NNodes', 'NCPUS',
                           ], axis=1, inplace=True)
    df_slurm['ReqBilling'] = df_slurm['ReqTRES'].str.split(',').str[0].str.split('=').str[1].astype(float)
    df_slurm.drop(columns=['ReqTRES'], inplace=True)
    df_slurm['AllocBilling'] = df_slurm['AllocTRES'].str.split(',').str[3].str.split('=').str[1].astype(float)
    df_slurm['AllocMem'] = df_slurm['AllocTRES'].str.split(',').str[2].str.split('=').str[1]
    df_slurm.drop(columns=['AllocTRES'], inplace=True)
    df_slurm.drop(labels=['NodeList'], axis=1, inplace=True)
    df_slurm.drop(columns=['State', 'DerivedExitCode', 'Reason'], inplace=True)

    exit_other = df_slurm['ExitCode'].value_counts()[df_slurm['ExitCode'].value_counts() < 100].index.to_list()
    df_slurm['ExitCode'] = df_slurm.ExitCode.apply(lambda x: 'other' if x in exit_other else x)
    df_slurm.Flags.replace(np.nan, 'NoFlag', inplace=True)

    df_slurm['ReqMem_in_MB'] = df_slurm.ReqMem.apply(convert_to_mb).astype(float)
    df_slurm.drop(labels=['ReqMem'], axis=1, inplace=True)
    df_slurm['Reserved_in_s'] = pd.to_timedelta(df_slurm.Reserved).dt.total_seconds().astype(int)
    df_slurm.drop(labels=['Reserved'], axis=1, inplace=True)
    df_slurm.Submit = pd.to_datetime(df_slurm.Submit)
    df_slurm.TimelimitRaw = df_slurm.TimelimitRaw.replace('Partition_Limit', 'nan')
    df_slurm.TimelimitRaw = df_slurm.TimelimitRaw.astype(float)
    df_slurm.AllocMem.fillna('nan', inplace=True)
    df_slurm.AllocMem.unique()
    df_slurm['AllocMem_in_MB'] = df_slurm.AllocMem.apply(convert_to_mb).astype(float)
    df_slurm.drop(labels=['AllocMem'], axis=1, inplace=True)
    df_slurm.drop('AllocBilling', axis=1, inplace=True)
    df_slurm["Start_day_of_week"] = df_slurm.Start.dt.weekday
    df_slurm["Start_hour_of_day"] = df_slurm.Start.dt.hour
    df_slurm["Start_month"] = df_slurm.Start.dt.month
    df_slurm.drop('Start', axis=1, inplace=True)
    df_slurm.dropna(axis=0, inplace=True)

    return df_slurm

        