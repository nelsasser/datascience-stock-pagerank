import datetime
import json

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

    corr_mat = np.corrcoef(time_series_data)
    corr_mat *= np.ones(corr_mat.shape)-np.diag(np.ones(corr_mat.shape[0]))

    if with_sharpe:
        sharpe_values = np.array([sharpe(ts_data,r) for ts_data in time_series_data])
        corr_mat *= sharpe_values[:, np.newaxis]

    if cap_zero:
        corr_mat = np.where(corr_mat >= 0, corr_mat, 0)

    return corr_mat

def page_rank(adj_mat):
    adj_mat = adj_mat/np.sum(np.abs(adj_mat), axis=0)
    n = len(adj_mat)
    r = np.repeat(1 / n, n)
    for t in range(50):
        r = np.matmul(adj_mat, r)
    return r

def graph_part(adj_mat, k, kmeans_iters=50):
    def fit_kmeans(k, data, iters=50):
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

def get_ticker_data(client, tickers, start, end, pt, ft, f):
    a = []
    for ticker in tickers:
        a.append(client.get_historical_price(
            ticker=ticker,
            start_date=start,
            end_date=end,
            period_type=pt,
            frequency_type=ft,
            frequency=f
        ))
    
    return pd.concat(a)

def get_returns(asset_data):
    asset_price_returns = asset_data[['open', 'close']].stack().diff()
    asset_percent_returns = (asset_price_returns / asset_data[['open', 'close']].stack().shift()).dropna()[:-1]

    return asset_price_returns, asset_percent_returns

def get_returns_for_all(client, universe, start_date, end_date, window_size, period_type, frequency_type, frequency):
    previous_trade_date = None
    trade_date = start_date
    lookback_date = trade_date - datetime.timedelta(days=window_size)

    current_data = get_ticker_data(client, universe, lookback_date, trade_date, period_type, frequency_type, frequency)

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
                daily_returns.append(daily_returns[-1])
            trade_date = trade_date + datetime.timedelta(days=1)
            continue
            
        # 1) Calculate daily profit and loss since last trade date, update current positions and position history
        if previous_trade_date is not None:
            print(trade_date, end=' ')
            print("Current positions", current_positions)   
            pl_data = get_ticker_data(client, [cp[0] for cp in current_positions], previous_trade_date, trade_date, period_type, frequency_type, frequency)

            # go through day by day for each stock
            # get the close position for that day, daily pnl for that stock is ([close] - [position open]) * [position volume]
            # pretty sure I am calculating this wrong / there is an easier way to do it with pandas though
            for dt in range((trade_date - previous_trade_date).days):
                daily_pnl = 0
                for pos_tck, pos_val, pos_vol, pos_dir in current_positions:
                    filtered = pl_data[ (pl_data['ticker'] == pos_tck) & 
                                        (pl_data['datetime'] == (previous_trade_date + datetime.timedelta(days=dt)))]
                    daily_pnl += np.sum(filtered['close'] - pos_val) * pos_vol * pos_dir
                
                balance += daily_pnl
                daily_returns.append(daily_pnl)

            # go through and update current positions and position history
            print('Closing:', end=' ')
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

                print((pos_tck, pos_opn, pos_vol, pos_dir), end=', ')

            print('')
            print('Balance', balance)
        
        # closed all current positions
        current_positions = []

        current_tickers = universe if selection_size is None else np.random.choice(list(universe), size=selection_size, replace=False)

        # 2) aggregate data
        current_data = get_ticker_data(client, current_tickers, lookback_date, trade_date, period_type, frequency_type, frequency)

        index_data = get_ticker_data(client, (index_asset, ), lookback_date, trade_date, period_type, frequency_type, frequency)
        _, index_percent_returns = get_returns(index_data)

        # for now pick the 10 stocks with highest heuristic
        tickers = []
        for ticker in current_tickers:
            ticker_data = current_data[current_data['ticker'] == ticker]
            _, ticker_percent_returns = get_returns(ticker_data)

            # is it sacreligious to make the average return of a stock the risk free rate?
            ticker_score = heuristic(ticker_percent_returns, np.mean(index_percent_returns))

            tickers.append((ticker, ticker_score))
        
        tickers.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0.0, reverse=True)
        top_tickers = tickers[:10]
        tickers, weights = [t[0] for t in top_tickers], np.array([t[1] for t in top_tickers]) / np.sum([t[1] for t in top_tickers])

        # allocate all of our capital to these stocks
        weights = weights * balance * alpha
        weights[np.where(np.isnan(weights))] = 0

        closing_prices = current_data[current_data['datetime'] == trade_date][['ticker', 'close']]
        for i, ticker in enumerate(tickers):
            price = closing_prices[closing_prices['ticker'] == ticker]['close'].values

            if len(price) == 0:
                vol = 0.0
                price = 0.0
                direction = 0.0
            else:
                price = price[0]
                vol = weights[i] / price
                direction = np.sign(weights)[i]
                
            current_positions.append((ticker, price, vol, direction))

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
            
    pl_data = get_ticker_data(client, [cp[0] for cp in current_positions], previous_trade_date, trade_date, period_type, frequency_type, frequency)

    # go through day by day for each stock
    # get the close position for that day, daily pnl for that stock is ([close] - [position open]) * [position volume]
    # pretty sure I am calculating this wrong / there is an easier way to do it with pandas though
    for dt in range((trade_date - previous_trade_date).days + 1):
        daily_pnl = 0
        for pos_tck, pos_val, pos_vol, pos_dir in current_positions:
            filtered = pl_data[ (pl_data['ticker'] == pos_tck) & 
                                (pl_data['datetime'] == (previous_trade_date + datetime.timedelta(days=dt)))]
            daily_pnl += np.sum(filtered['close'] - pos_val) * pos_vol
        
        balance += daily_pnl
        daily_returns.append(daily_pnl)

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
        universe = list(universe)[:100]

        client = TDAPI(api_key=api_key)
        client.login(driver_path=chrome_driver)

        start_date = datetime.datetime.strptime('2018-01-01', '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime('2020-12-31', '%Y-%m-%d').date()

    else:
        from testclient import TestClient

        # top 10 market cap S&P 500 tickers (as of april 2021)
        universe = set([t.strip() for t in open('./data/tickers.txt').readlines()])

        client = TestClient()

        # reduced start and end date times
        start_date = datetime.datetime.strptime('2018-01-1', '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime('2018-12-31', '%Y-%m-%d').date()
    
    window = 5                  # look at 5 days at a time
    select = None               # number of stocks to randomly choose from universe for each backtest step, set to None to do all
    ts = 5                      # how many days to step forward each iteration
    pt = client.PeriodType.YEAR        # period to aggregate (matters for tda api)
    ft = client.FrequencyType.DAILY    # time scale we want to aggregate by
    f = client.Frequency.DAILY         # how much to aggregate
    alpha = 0.01                        # controls how much of our balance we allocate per timeframe


    # basic heuristic function, calculates sharpe of x against y with a floor of 0 so everything stays positive
    h_func = lambda x, y: max(0, sharpe(x, y))

    # asset to use as index when calculating metrics
    # using SPY since it tracks S&P 500 and is popular
    index_asset = 'SPY'

    client.load(universe, index_asset)
    end_bal, positions, returns = backtest( client=client, 
                                            universe=universe,
                                            index_asset=index_asset,
                                            selection_size=select, 
                                            start_date=start_date + datetime.timedelta(days=window), 
                                            end_date=end_date, 
                                            window_size=window, 
                                            timestep=ts, 
                                            heuristic=h_func, 
                                            period_type=pt, 
                                            frequency_type=ft, 
                                            frequency=f,
                                            alpha=alpha)

    import matplotlib.pyplot as plt
    x = np.arange(len(returns))
    returns = np.array(returns)
    returns[np.where(np.isnan(returns))] = 0
    plt.plot(x, returns)
    plt.show()

    cum_returns = np.cumsum(returns)
    plt.plot(x, cum_returns)
    plt.show()
