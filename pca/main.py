
import yfinance as yf
import pandas as pd
import numpy as np
import pickle

import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def getSP500():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

    df_sp500 = table[0]

    ticker_names = str(df_sp500['Symbol'].values).replace("'", "").replace("[", "").replace("]", "").split()  

    return ticker_names



if __name__ == '__main__':

    print(f"Running main.py: __name__ = {__name__} and __package__ = {__package__}")
    print(f"Current working directory: {os.getcwd()}")
    year_selected = 2024
    local_cached_data = os.path.join(os.getcwd(), 'pca', 'data.npy')
    print(f"Local cached data: {local_cached_data}")

    ticker_names = getSP500()

    if os.path.isfile(local_cached_data):
        X = np.load(local_cached_data)
    else:
        data = yf.download(ticker_names, start="2024-01-01", end="2024-12-31")
        df_close = data['Adj Close'].dropna(axis='columns', how='all')
        col_names = df_close.to_csv('data.csv')
        X = df_close.to_numpy()
        np.save('data.npy', X)

    print(X)   

    # Find columns with NaN
    columns_with_nan = np.isnan(X).any(axis=0)

    # Get data without the columns containing NaN
    data_clean = X[:, ~columns_with_nan] 
    df_close = pd.DataFrame(data_clean)


    X = StandardScaler().fit_transform(data_clean)

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)

    print("Explained Variance of Each Component:" )
    print(pca.explained_variance_ratio_)

    # get sum of weights for first 3 principle components
    stock_weights_pca = pca.components_[0,:] + pca.components_[1,:] + pca.components_[2,:]
    # define number of stocks to include in index fund
    top_n = 61

    # get boolean array of top n elements 
    bool_top_stocks = abs(stock_weights_pca) > np.sort(abs(stock_weights_pca))[len(stock_weights_pca)-top_n]

    # create data structures with weights and ticker names that define index fund
    index_fund_weights = (stock_weights_pca[bool_top_stocks])
    index_fund_tickers = df_close.columns[bool_top_stocks]

    import matplotlib.pyplot as plt

    # create figure
    plt.figure(num=None, figsize=(24, 6))

    plt.bar(np.arange(len(index_fund_weights)), index_fund_weights/np.max((index_fund_weights)), color = 'c', edgecolor = 'k')
    plt.title("Relative Stock Weights in Index Fund", fontsize=16)
    plt.xticks(np.arange(len(index_fund_weights)), index_fund_tickers, rotation=90)
    plt.xlabel('Stock Ticker Names', fontsize=12)
    plt.ylabel('Relative Weight', fontsize=12)
    plt.show

    plt.figure(num=None, figsize=(24, 6))

    # plot S&P 500 over time
    plt.subplot(2, 1, 1)
    plt.plot(np.sum(df_close, axis=1))
    plt.title("S&P 500 (2023)", fontsize=16)

    # plot index fund over time
    plt.subplot(2, 1, 2)
    plt.plot(-1*np.sum(index_fund_weights * df_close.iloc[:,bool_top_stocks], axis=1))
    plt.title("PCA Derived Index Fund (2023)", fontsize=16)

    # add space between plots
    plt.subplots_adjust(hspace = 0.5 )

    # actual percent return of S&P500 
    actual_percent_return = np.sum((df_close.iloc[len(df_close)-1,:] - df_close.iloc[0,:]))/np.sum((df_close.iloc[0,:]))

    print('Percent return of S&P 500:')
    print(actual_percent_return)

    top_stocks_pc1_percent_return = np.sum(index_fund_weights * (df_close.iloc[len(df_close)-1,bool_top_stocks] - df_close.iloc[0,bool_top_stocks]))/np.sum(index_fund_weights * (df_close.iloc[0,bool_top_stocks]))

    print('Percent return of PCA derived index fund:')
    print(top_stocks_pc1_percent_return)
