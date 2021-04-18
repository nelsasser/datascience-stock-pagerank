from main import *


if __name__ == '__main__':
    # change this to false use test data
    use_api = False

    if use_api:
        from api import TDAPI

        data = json.load(open('./key.json'))
        api_key, chrome_driver = data['api_key'], data['driver_path']

        # all S&P 500 tickers (as of april 2021)
        universe = set([t.strip() for t in open('./tickers.txt').readlines()])

        client = TDAPI(api_key=api_key)
        client.login(driver_path=chrome_driver)

        start_date = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime('2020-12-31', '%Y-%m-%d').date()

    else:
        from testclient import TestClient

        # top 10 market cap S&P 500 tickers (as of april 2021)
        universe = set([t.strip() for t in open('./data/tickers.txt').readlines()])

        client = TestClient()
        client.load(fltr=[])  # put tickers you don't want to load in here to speed things up

        # reduced start and end date times
        start_date = datetime.datetime.strptime('2018-01-1', '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime('2018-12-31', '%Y-%m-%d').date()

    window = 100  # look at 5 days at a time
    select = None  # number of stocks to randomly choose from universe for each backtest step, set to None to do all
    ts = 5  # how many days to step forward each iteration
    pt = client.PeriodType.YEAR  # period to aggregate (matters for tda api)
    ft = client.FrequencyType.DAILY  # time scale we want to aggregate by
    f = client.Frequency.DAILY  # how much to aggregate

    # basic heuristic function, calculates sharpe of x against y with a floor of 0 so everything stays positive
    h_func = lambda x, y: sharpe(x, y)

    # asset to use as index when calculating metrics
    # using SPY since it tracks S&P 500 and is popular
    index_asset = 'SPY'

    res = get_returns_for_all(  client=client,
                                universe=universe,
                                start_date=start_date + datetime.timedelta(days=window),
                                end_date=end_date,
                                window_size=window,
                                period_type=pt,
                                frequency_type=ft,
                                frequency=f)

    print(res)