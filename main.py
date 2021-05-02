import datetime
import json
import networkx as nx
import matplotlib.pyplot as plt

from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.tsa.stattools as ts
from scipy.spatial.distance import squareform, pdist

import pandas as pd
import numpy as np

def sharpe(x, r):
    """Get sharpe ratio for x

    Args:
        x (numpy.array): array of returns data
        r (float): risk free rate

    Returns:
        float: sharpe ratio
    """
    if len(x.shape) == 1:
        expected_return = np.mean(x)
        return_deviation = np.std(x)
    else:
        expected_return = np.mean(x)
        return_deviation = np.std(x)

    return (expected_return - r) / return_deviation

def beta(x, y):
    """Get beta coefficient for x compared against y

    Args:
        x (np.array): asset returns
        y (np.array): index returns

    Returns:
        float: beta coefficient
    """

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov = np.sum((x - x_mean) * (y - y_mean)) / (x.shape[0] - 1)
    var = np.var(x)

    return cov/var

def data_to_adj_mat(time_series_data, with_sharpe, cap_zero, r):
    # calc_cov_weight = lambda x, y: coint_johansen(np.array([x, y]).T, 0, 1).max_eig_stat
    # ts.coint(time_series_data[0], time_series_data[1])
    """Converts time series data into a 2d matrix of relationships between stocks

        Args:
            time_series_data (pandas dataframe): list of values of stock prices over time
            with_sharpe (bool): boolean flag to incorporate the sharpe ratio
            cap_zero (bool): boolean flag to cap the minimum value at zero
            r (float): risk free rate

        Returns:
            2d list: adjacency matrix
    """

    corr_mat = np.corrcoef(time_series_data, rowvar=False)
    corr_mat *= np.ones(corr_mat.shape)-np.diag(np.ones(corr_mat.shape[0]))

    if with_sharpe:
        sharpe_values = sharpe(time_series_data, r)
        corr_mat = (np.abs(corr_mat.T) * sharpe_values.to_numpy()).T

    if cap_zero:
        corr_mat = np.where(corr_mat >= 0, corr_mat, 0)

    return corr_mat

def page_rank(adj_mat):
    """Runs the page rank algorithm on the graph of stock relationships

        Args:
            adj_mat (2d list): correlation values from one stock to another

        Returns:
            list: value/power of each stock
    """
    adj_mat = adj_mat/np.sum(np.abs(adj_mat), axis=0)
    n = len(adj_mat)
    r = np.repeat(1 / n, n)
    for t in range(50):
        r = np.matmul(adj_mat, r)
        r /= np.sum(np.abs(r))
    return r

def graph_part(adj_mat, k, kmeans_iters=50):
    """
    Creates 'k' groups of the graph with the adjacency matrix `adj_mat`.
    We then run graph partitioning on the graph.

    :param adj_mat: adjacency matrix
    :param k: number of partitions
    :param kmeans_iters: max iterations to run k-means
    :return:
    """
    def fit_kmeans(k, data, iters=50):
        """
        internal k-means function

        :param k: number of groups to fund
        :param data: np.ndarray with shape nxm: n data points m dimensions per data point
        :param iters: max iters
        :return:
        """
        p = np.random.permutation(data.shape[0])
        centriods = data[p[:k]]

        dists = np.zeros((k, data.shape[0]))

        for iter_num in range(iters + 1):
            for m in range(k):
                dists[m, :] = np.linalg.norm(data - centriods[m], axis=1)

            classes = np.argmin(dists, axis=0)

            old_centriods = centriods.copy()

            if iter_num < iters:
                for m in range(k):
                    if np.sum(classes == m) != 0:
                        # calc new centers
                        centriods[m] = np.dot((classes == m), data)
                        centriods[m] /= np.sum(classes == m)

                if np.allclose(old_centriods, centriods):
                    break

        return centriods, classes

    D = np.diag(np.sum(adj_mat, axis=0))
    L = D - adj_mat
    eig_vals, eig_vecs = np.linalg.eigh(L)
    p = np.argsort(eig_vals)

    fiedler = np.array(eig_vecs[:, p[1]]).ravel()

    classes = fit_kmeans(k, fiedler[:, np.newaxis], iters=kmeans_iters)[1]

    return classes

def plot_partitionedGraph(classes, universe,adj_mat):
    """Creates and plots a graph using networkx to show graph partitioning results and correlation via edge weights/colorings

        Args:
            classes (list): stock groups
            universe (list): corresponding int index value to stock name
            adj_mat (2d list): correlation values from one stock to another

        Returns:
            nothing
    """
    G = nx.Graph()
    temp = sorted(universe)
    for i in range(len(temp)):
        G.add_node(i)
    weights = []
    edge_list =[]
    for i in range(len(temp)):
        for j in range(len(temp)):
            G.add_edge(i,j)
            rgb = (1-adj_mat[i][j]**2,1-adj_mat[i][j]**2,1-adj_mat[i][j]**2)
            weights.append(rgb)
            edge_list.append((i,j))

    labels = dict()
    for i in range(len(temp)):
        labels[i] = temp[i]
    np.random.seed(4022)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=classes, cmap='Pastel1', node_size=500,alpha=.9)
    nx.draw_networkx_edges(G, pos,edge_color=weights, edge_cmap=plt.cm.binary, edgelist=edge_list)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    plt.axis("off")
    plt.show()




def get_returns(asset_data, close_only=False):
    if close_only:
        reordered_data = asset_data[['close', 'datetime', 'ticker']].set_index(['ticker', 'datetime']).stack().unstack(level=0)
    else:
        reordered_data = asset_data[['open', 'close', 'datetime', 'ticker']].set_index(['ticker', 'datetime']).stack().unstack(level=0)
    asset_price_returns = reordered_data.diff()
    asset_percent_returns = (asset_price_returns / reordered_data.shift()).dropna()[:-1]

    return asset_price_returns, asset_percent_returns

def get_returns_for_all(client, universe, start_date, end_date, window_size, period_type, frequency_type, frequency):
    previous_trade_date = None
    trade_date = start_date
    lookback_date = trade_date - datetime.timedelta(days=window_size)

    current_data = client.get_historical_price(universe, lookback_date, trade_date, period_type, frequency_type, frequency)

    ticker_returns = []
    for ticker in universe:
        ticker_data = current_data[current_data['ticker'] == ticker]
        _, ticker_percent_returns = get_returns(ticker_data)
        ticker_returns.append(list(ticker_percent_returns))

    ticker_returns_as_array = np.array(ticker_returns)
    assert len(ticker_returns_as_array.shape) == 2

    return ticker_returns_as_array

def check_day(client, dt):
    df = client.get_historical_price('SPY', dt, dt, client.PeriodType.YEAR, client.FrequencyType.DAILY, client.Frequency.DAILY)
    return df.empty

def backtest(client, universe, index_asset, selection_size, start_date, end_date, window_size, timestep, heuristic, period_type, frequency_type, frequency, alpha):
    # big main idea is to avoid looking into the future
    # for each trade date we buy using the close price and sell using open price
    # for calculating metrics we need to be sure we ONLY USE THE OPEN value of the trade date. 
    # NEVER USE THE CLOSE VALUE OF THE TRADE DATE FOR CALCULATING METRICS
    
    previous_trade_date = None
    trade_date = start_date
    lookback_date = trade_date - datetime.timedelta(days=window_size)

    balance = 1_000_000 # starting amount of money

    current_positions = []
    position_history = [] # position is (ticker, price, volume, direction)
    daily_returns = []

    # run backtest
    while(trade_date <= end_date):
        if check_day(client, trade_date):
            if len(daily_returns):
                daily_returns.append((trade_date, 0.0))
            trade_date = trade_date + datetime.timedelta(days=1)
            continue
            
        # 1) Calculate daily profit and loss since last trade date, update current positions and position history
        if previous_trade_date is not None:
            print(trade_date, end=' ')
            if len(current_positions) == 0:
                print("Current positions NONE")
                for dt in range((trade_date - previous_trade_date).days):
                    daily_returns.append((trade_date + datetime.timedelta(days=dt), 0.0))
            else:
                print("Current positions", len(current_positions))   
                pl_data = client.get_historical_price(ticker=[cp[0] for cp in current_positions], 
                                                      start_date=previous_trade_date, 
                                                      end_date=trade_date, 
                                                      period_type=period_type, 
                                                      frequency_type=frequency_type, 
                                                      frequency=frequency)

                # go through day by day for each stock
                # get the close position for that day, daily pnl for that stock is ([close] - [position open]) * [position volume]
                current_tickers = [c[0] for c in current_positions]
                filtered_pl = pl_data[  (pl_data['ticker'].isin(current_tickers)) & \
                                        (pl_data['datetime'] < trade_date) & \
                                        (pl_data['datetime'] >= previous_trade_date)]

                cp_dct = {x[0]: (x[1], x[2] * x[3]) for x in current_positions}
                pnl = (filtered_pl['close'] - filtered_pl['ticker'].map(lambda x: cp_dct[x][0])) * filtered_pl['ticker'].map(lambda x: cp_dct[x][1])

                pnl.index = pd.MultiIndex.from_arrays([filtered_pl['ticker'].copy(), filtered_pl['datetime'].copy()], names=('ticker', 'datetime'))

                # mask = np.ones(len(set(filtered_pl['ticker'])))
                # v = np.abs(pnl.groupby('ticker').sum().index.map(lambda x: cp_dct[x][0]) * pnl.groupby('ticker').sum().index.map(lambda x: cp_dct[x][1]))
                
                # pnl = pnl.unstack(level=0)
                # ppl = pnl / v

                # # cut trades early if they have a current cumulative loss >= 10%
                # pnl *= ~((~(ppl > -0.025).shift().fillna(True)).cumsum().astype(bool))

                # pnl = pnl.stack()
                daily_pnl = pnl.groupby('datetime').sum().to_frame('pnl')
                    
                balance += np.sum(daily_pnl['pnl'].to_numpy())
                rec = daily_pnl.reset_index().to_records(index=False)
                daily_returns += list(rec)

                # go through and update current positions and position history
                # print('Closing:', end=' ')
                for pos_tck, pos_val, pos_vol, pos_dir in current_positions:
                    # get tickers open price for today
                    pos_data = pl_data[(pl_data['ticker'] == pos_tck) & (pl_data['datetime'] == trade_date)]
                    pos_opn = pos_data['open'].values

                    if len(pos_opn) == 0:
                        pos_opn = 0.0
                    else:
                        pos_opn = pos_opn[0]

                    # update the position open and close in the position history
                    position_history.append([(pos_tck, pos_val, pos_vol, pos_dir), (pos_tck, pos_opn, pos_vol, pos_dir)])

                    # print((pos_tck, pos_opn, pos_vol, pos_dir), end=', ')

                # print('')
            print('Balance', balance)
        
        # closed all current positions
        current_positions = []

        current_tickers = list(universe) if selection_size is None else list(np.random.choice(list(universe), size=selection_size, replace=False))

        # 2) aggregate data
        current_data = client.get_historical_price(ticker=current_tickers + [index_asset], 
                                                   start_date=lookback_date, 
                                                   end_date=trade_date, 
                                                   period_type=period_type, 
                                                   frequency_type=frequency_type, 
                                                   frequency=frequency)

        _, ticker_returns = get_returns(current_data)
        avg_index_returns = np.mean(ticker_returns[index_asset])

        # is it sacreligious to make the average return of a stock the risk free rate?
        ticker_scores = pd.Series(index=ticker_returns.columns, data=heuristic(ticker_returns))

        # ticker_scores = heuristic(ticker_returns)
        # select_ticker_indexes = np.abs(ticker_scores) > 0.5
        # select_ticker_weights = ticker_scores[select_ticker_indexes] / np.sum(np.abs(ticker_scores[select_ticker_indexes]))

        select_ticker_indexes = np.abs(ticker_scores) > np.mean(np.abs(ticker_scores))
        select_ticker_weights = ticker_scores[select_ticker_indexes] / np.sum(ticker_scores[select_ticker_indexes])
        if len(select_ticker_weights) > 0:
            assert(abs(1 - np.sum(select_ticker_weights)) < 1e-6)

        # allocate all of our capital to these stocks
        weights = select_ticker_weights * balance * alpha

        closing_prices = current_data[current_data['datetime'] == trade_date][['ticker', 'close']]
        for ticker in select_ticker_indexes.index[select_ticker_indexes]:
            # ticker = ticker_data.columns[select_ticker_indexes[i]]
            price = closing_prices[closing_prices['ticker'] == ticker]['close'].values

            if len(price) == 0:
                continue

            price = price[0]
            vol = weights[ticker] / price
            direction = np.sign(weights)[ticker]

            if abs(vol) < 1e-3:
                continue

            current_positions.append((ticker, price, abs(vol), direction))

        # 4) calculate heuristics and construct M
        # 5) run pagerank power iteration to get r
        # 6) calculate new current positions from r and balance
        
        # 7) update window
        previous_trade_date = trade_date
        trade_date = trade_date + datetime.timedelta(days=timestep)
        lookback_date = trade_date - datetime.timedelta(days=window_size)

    # sell current holdings and calculate final PL
    while(check_day(client, trade_date)):
        trade_date = trade_date - datetime.timedelta(days=1)
        continue

    if len(current_positions) == 0:
        print("Current positions NONE")
        for dt in range((trade_date - previous_trade_date).days):
            daily_returns.append((trade_date + datetime.timedelta(days=dt), 0.0))
    else:    
        pl_data = client.get_historical_price(ticker=[cp[0] for cp in current_positions], 
                                              start_date=previous_trade_date, 
                                              end_date=trade_date, 
                                              period_type=period_type, 
                                              frequency_type=frequency_type, 
                                              frequency=frequency)

        # go through day by day for each stock
        # get the close position for that day, daily pnl for that stock is ([close] - [position open]) * [position volume]
        # pretty sure I am calculating this wrong / there is an easier way to do it with pandas though
        filtered_pl = pl_data[  (pl_data['ticker'].isin([c[0] for c in current_positions])) & \
                                (pl_data['datetime'] < trade_date) & \
                                (pl_data['datetime'] >= previous_trade_date)]

        cp_dct = {x[0]: (x[1], x[2] * x[3]) for x in current_positions}
        pnl = (filtered_pl['close'] - filtered_pl['ticker'].map(lambda x: cp_dct[x][0])) * filtered_pl['ticker'].map(lambda x: cp_dct[x][1])
        pnl.index = filtered_pl['datetime']
        daily_pnl = pnl.groupby('datetime').sum().to_frame('pnl')
        
        balance += np.sum(daily_pnl['pnl'].to_numpy())
        daily_returns += list(daily_pnl.reset_index().to_records(index=False))

        # go through and update current positions and position history
        for pos_tck, pos_val, pos_vol, pos_dir in current_positions:
            # get tickers open price for today
            pos_data = pl_data[(pl_data['ticker'] == pos_tck) & (pl_data['datetime'] == trade_date)]
            pos_opn = pos_data['open'].values[0]

            # update the position open and close in the position history
            position_history.append([(pos_tck, pos_val, pos_vol, pos_dir), (pos_tck, pos_opn, pos_vol, pos_dir)])

    return balance, position_history, daily_returns
    # from balance_history calculate the sharpe, beta, etc whatever metrics we want to use against our index
    # return backtest metrics

if __name__ == '__main__':
    # change this to false use test data
    use_api = True

    if use_api:
        from api import TDAPI            
        
        data = json.load(open('./key.json'))
        api_key, chrome_driver = data['api_key'], data['driver_path']

        # all S&P 500 tickers (as of april 2021)
        universe = set([t.strip() for t in open('./tickers.txt').readlines()])
        universe = list(universe)

        client = TDAPI(api_key=api_key)
        client.login(driver_path=chrome_driver)

        start_date = datetime.datetime.strptime('2012-01-01', '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime('2020-12-31', '%Y-%m-%d').date()

    else:
        from testclient import TestClient

        # top 10 market cap S&P 500 tickers (as of april 2021)
        universe = set([t.strip() for t in open('./data/tickers.txt').readlines()])

        client = TestClient()

        # reduced start and end date times
        start_date = datetime.datetime.strptime('2018-01-1', '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime('2018-12-31', '%Y-%m-%d').date()
    
    window = 5                          # look at 5 days at a time
    select = None                       # number of stocks to randomly choose from universe for each backtest step, set to None to do all
    ts = 5                              # how many days to step forward each iteration
    pt = client.PeriodType.YEAR         # period to aggregate (matters for tda api)
    ft = client.FrequencyType.DAILY     # time scale we want to aggregate by
    f = client.Frequency.DAILY          # how much to aggregate
    alpha = 0.2                         # controls how much of our balance we allocate per timeframe

    # basic heuristic function, calculates sharpe of x against y with a floor of 0 so everything stays positive
    # h_func = lambda x, y: max(0, sharpe(x, y))
    def proprietary_sauce(ticker_returns):
        return sharpe(ticker_returns, 0.0)

    # asset to use as index when calculating metrics
    # using SPY since it tracks S&P 500 and is popular
    index_asset = 'SPY'

    client.load(universe, index_asset)
    print('Starting backtest')
    end_bal, positions, daily_returns = backtest( client=client, 
                                            universe=universe,
                                            index_asset=index_asset,
                                            selection_size=select, 
                                            start_date=start_date + datetime.timedelta(days=window), 
                                            end_date=end_date, 
                                            window_size=window, 
                                            timestep=ts, 
                                            heuristic=proprietary_sauce, 
                                            period_type=pt, 
                                            frequency_type=ft, 
                                            frequency=f,
                                            alpha=alpha)

    import matplotlib.pyplot as plt

    daily_returns = pd.DataFrame.from_records(daily_returns, columns=['datetime', 'returns'])
    daily_returns['datetime'] = pd.to_datetime(daily_returns['datetime']).dt.date
    daily_returns = daily_returns.groupby(by='datetime').mean()

    returns = daily_returns['returns'].to_numpy()
    days = list(daily_returns.index)

    x = np.arange(len(returns))
    returns = np.array(returns)
    returns[np.where(np.isnan(returns))] = 0

    p_returns = daily_returns['returns'].fillna(0) / (daily_returns['returns'].shift().fillna(0).cumsum() + 1000000)
    
    index_returns = get_returns(client.data[(client.data['ticker'] == index_asset) & \
                                                        (client.data['datetime'] >= days[0]) & \
                                                        (client.data['datetime'] <= days[-1] + datetime.timedelta(days=1))][['ticker', 'datetime', 'close']], True)[1][1:]
    index_returns = index_returns.reset_index().drop(columns='level_1')
    index_returns['datetime'] = index_returns['datetime'].dt.date
    index_returns = index_returns.set_index('datetime', drop=True)

    daily_returns['returns'] = p_returns
    daily_returns['input_returns'] = returns / 1_000_000
    compare_returns = daily_returns.join(index_returns, how='inner', on='datetime')

    wins = []
    losses = []
    shorts = []
    longs = []
    for trade in positions:
        trade_open, trade_close = trade[0], trade[1]

        if trade_open[1] < trade_close[1]:
            if trade_open[3] == 1:
                wins.append(trade)
                longs.append((1, trade))
            else:
                losses.append(trade)
                shorts.append((0, trade))
        elif trade_open[1] > trade_close[1]:
            if trade_open[3] == -1:
                wins.append(trade)
                shorts.append((1, trade))
            else:
                losses.append(trade)
                longs.append((0, trade))

    print(f'\n\nWin %: {round((len(wins) / (len(wins) + len(losses))) * 100, 2)}% --- {len(wins)}/{len(losses)}/{len(wins+losses)}')
    print('Top 5 losses:')
    losses.sort(key=lambda x: (x[1][1] - x[0][1]) * x[0][2] * x[0][3])
    for i in range(5):
        print(f'  {i + 1}) {losses[i][0][0]} - {losses[i][0][1]} > {losses[i][1][1]} @ {losses[i][0][2]} = {(losses[i][1][1] - losses[i][0][1]) * losses[i][0][2] * losses[i][0][3]}')
    
    print(f'\nLong/Short ratio {round(len(longs) / len(shorts), 2)} -- {len(longs)}/{len(shorts)}')
    print(f'Long win %: {round(np.sum([l[0] == 1 for l in longs]) / len(longs), 2) * 100}%')
    print(f'Short win %: {round(np.sum([s[0] == 1 for s in shorts]) / len(shorts), 2) * 100}%')
    
    shrp = sharpe(compare_returns['returns'].to_numpy(), np.mean(compare_returns[index_asset].to_numpy()))
    print(f'\nSharpe: {shrp}')

    shrp_0 = sharpe(compare_returns['returns'].to_numpy(), 0.0)
    print(f'\nSharpe (0): {shrp_0}')

    B = beta(compare_returns['returns'].to_numpy(), compare_returns[index_asset].to_numpy())
    print(f'\nBeta: {B}')

    fig, axes = plt.subplots(1, 3)
    axes[0].plot(x, returns)
    axes[0].set_title('Daily Returns over Time')
    
    axes[1].hist(returns, bins=100)
    axes[1].set_title('Histogram of Daily Returns')
    
    compare_returns['input_returns'] = np.cumsum(compare_returns['input_returns'])
    compare_returns[index_asset] = np.cumsum(compare_returns[index_asset])
    compare_returns[['input_returns', index_asset]].plot(ax=axes[2])
    axes[2].set_title('Returns v.s. SPY')
    
    plt.show()
    print('')
