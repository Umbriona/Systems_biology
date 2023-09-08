import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calc_all(df, max_distance = 8.0, name="blabla"):

    df.columns = df.index
    for col in df.columns:
        df[col].values[df[col]>= max_distance] = 0
        df[col].values[df[col]> 0] = 1
    
    net = nx.from_pandas_adjacency(df)

    degree_list = [d for n, d in net.degree]
    eigenvector_list = [d for n, d in nx.eigenvector_centrality(net).items()]
    betweenness_list = [d for n, d in nx.betweenness_centrality(net).items()]
    closeness_list = [d for n, d in nx.closeness_centrality(net).items()]
    ids = [name for _ in degree_list]
    df_central = pd.DataFrame({"id":ids, "Degree":degree_list, "Eigenvector":eigenvector_list, "Betweeness": betweenness_list, "Closeness": closeness_list})
    print(f"{np.mean(degree_list)}\t{np.mean(betweenness_list)}\t{np.mean(closeness_list)}\t{np.mean(eigenvector_list)}")
    return df_central

def calc_all_idx(df, max_distance = 8.0, name="blabla", idx=[0], type = "Var"):

    df.columns = df.index
    for col in df.columns:
        df[col].values[df[col]>= max_distance] = 0
        df[col].values[df[col]> 0] = 1
    
    net = nx.from_pandas_adjacency(df)

    degree_list = np.array([d for n, d in net.degree])[idx]
    eigenvector_list = np.array([d for n, d in nx.eigenvector_centrality(net).items()])[idx]
    betweenness_list = np.array([d for n, d in nx.betweenness_centrality(net).items()])[idx]
    closeness_list = np.array([d for n, d in nx.closeness_centrality(net).items()])[idx]
    ids = [name for _ in range(len(idx))]
    t = [type for _ in range(len(idx))]
    df_central = pd.DataFrame({"id":ids, "Degree":degree_list, "Eigenvector":eigenvector_list, "Betweeness": betweenness_list, "Closeness": closeness_list, "Class": t, "idx":idx})
    print(f"{np.mean(degree_list)}\t{np.mean(betweenness_list)}\t{np.mean(closeness_list)}\t{np.mean(eigenvector_list)}")
    return df_central
    