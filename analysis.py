import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df1 = pd.read_csv('./out.csv')
    df2 = pd.read_csv('./out2.csv')

    df = pd.DataFrame()
    df['SPY Returns'] = df1['SPY'] * 100
    df['Pagerank Returns'] = df1['input_returns'] * 100
    df['Naive Returns'] = df2['input_returns'] * 100
    df.index = df1['datetime']

    df.plot(title='Returns v.s. SPY', xlabel='Date', ylabel='Returns (%)')
    plt.show()