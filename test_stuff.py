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

        # reduced start and end date times
        start_date = datetime.datetime.strptime('2018-01-1', '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime('2018-12-31', '%Y-%m-%d').date()

    window = 100  # look at 100 days at a time
    pt = client.PeriodType.YEAR  # period to aggregate (matters for tda api)
    ft = client.FrequencyType.DAILY  # time scale we want to aggregate by
    f = client.Frequency.DAILY  # how much to aggregate

    # asset to use as index when calculating metrics
    # using SPY since it tracks S&P 500 and is popular
    index_asset = 'SPY'

    client.load(universe, index_asset)
    universe_list = list(universe)

    curr_data = client.get_historical_price(universe_list,
                                            start_date=start_date + datetime.timedelta(days=window),
                                            end_date=end_date,
                                            period_type=pt,
                                            frequency_type=ft,
                                            frequency=f)
    _, res = get_returns(curr_data)

    #
    adj_mat_with_sharpe = data_to_adj_mat(res, True, True, 0)

    # we need an undirected graph for graph_part so we only look at correlation
    adj_mat = data_to_adj_mat(res, False, True, 0)

    page_rank_res = page_rank(adj_mat_with_sharpe)
    p = np.argsort(-page_rank_res)
    print(np.array(universe_list)[p], page_rank_res[p])

    graph_part_res = graph_part(adj_mat, 5)
    for part_class in np.unique(graph_part_res):
        print(np.array(universe_list)[np.where(graph_part_res == part_class)[0]])

    plot_partitionedGraph(graph_part_res,universe_list,adj_mat)
