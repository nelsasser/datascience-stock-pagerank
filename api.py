import json
import datetime
import time

import pandas as pd
from tda import auth, client

class TDAPI:
    PeriodType = client.Client.PriceHistory.PeriodType
    Period = client.Client.PriceHistory.Period
    FrequencyType = client.Client.PriceHistory.FrequencyType
    Frequency = client.Client.PriceHistory.Frequency

    def __init__(self, api_key, token_path='./token.pickle', redirect_uri='https://localhost'):
        self.token_path = token_path
        self.api_key = api_key
        self.redirect_uri = redirect_uri
        self.tda_client = None
        self.last_call = None
        # self.rate_limiter = []

    def load(self, universe, index_asset):
        print("Loading data for backtest")

        try:
            self.data = pd.read_csv('D:\\stock_project\\all.csv')
            self.data['datetime'] = pd.to_datetime(self.data['datetime']).dt.date
        except:
            print("Failed to find local data, getting from API")
            d = []
            for i, ticker in enumerate(universe):
                print(f'{i}/{len(universe)} done.')
                # rate limit ourselves so we don't get hit with TDA rate limiter which forces us to pause for 1 minute
                self.rate_limit()

                # self.rate_limiter.append(datetime.datetime.now())

                res = self.tda_client.get_price_history(
                    ticker,
                    period_type=self.PeriodType.YEAR,
                    period=self.Period.TWENTY_YEARS,
                    frequency_type=self.FrequencyType.DAILY,
                    frequency=self.Frequency.DAILY
                )

                if res.status_code == 200:
                    res = pd.DataFrame(res.json()['candles'])
                    res['ticker'] = ticker
                    if not res.empty:
                        res['datetime'] = pd.to_datetime(res['datetime'], unit='ms')
                        res['datetime'] = res['datetime'].dt.date
                elif res.status_code == 429:
                    print("Rate limiter failed, pausing for 1 minute then trying again.")
                    time.sleep(60)
                    continue
                else:
                    print(f"Error could not get data for {ticker}: {res.status_code}")
                    res = pd.DataFrame()

                d.append(res)
            
            while(True):
                # rate limit ourselves so we don't get hit with TDA rate limiter which forces us to pause for 1 minute
                self.rate_limit()
                res = self.tda_client.get_price_history(
                    index_asset,
                    period_type=self.PeriodType.YEAR,
                    period=self.Period.TWENTY_YEARS,
                    frequency_type=self.FrequencyType.DAILY,
                    frequency=self.Frequency.DAILY
                )

                if res.status_code == 200:
                    res = pd.DataFrame(res.json()['candles'])
                    res['ticker'] = index_asset
                    if not res.empty:
                        res['datetime'] = pd.to_datetime(res['datetime'], unit='ms')
                        res['datetime'] = res['datetime'].dt.date
                    break
                elif res.status_code == 429:
                    print("Rate limiter failed, pausing for 1 minute then trying again.")
                    time.sleep(60)
                    continue
                else:
                    print(f"Error could not get data for {ticker}: {res.status_code}")
                    res = pd.DataFrame()
                    break

            d.append(res)
            self.data = pd.concat(d)
        
    def rate_limit(self, limit=120):
        min_time = 0.05 # basically error, get this as close to 0 as possible to not hit rate limiter since 0 sometimes still triggers it
        # per-call delay
        pcd = 60 / limit
        if self.last_call is None:
            wait_time = min_time
        else:
            wait_time = max(pcd - (time.time() - self.last_call), min_time) # (time we must wait) - (time we've already waited)

        print(f'Waiting {wait_time} sec')
        time.sleep(wait_time)
        self.last_call = time.time()

        # # get current time
        # cur_time = datetime.datetime.now()

        # # get rid of all calls that were more than 1 minute ago
        # self.rate_limiter = list(filter(lambda x: (cur_time - x).total_seconds() / 60.0 < 1, self.rate_limiter))
        
        # # number of available calls that we can make in the current elapsed
        # available = limit - len(self.rate_limiter)

        # if available:
        #     return
        # else:
        #     # sleep until room in queue
        #     # gets the length of the current elapsed minute and waits until it has finished
        #     t = 60 - (cur_time - self.rate_limiter[0]).total_seconds()/60
        #     print(f'Self waiting for {t} seconds')
        #     time.sleep(t)
        #     return

    def login(self, driver_path):
        try:
            self.tda_client = auth.client_from_token_file(self.token_path, self.api_key)
        except FileNotFoundError:
            from selenium import webdriver
            driver = webdriver.Chrome(executable_path=driver_path)
            self.tda_client = auth.client_from_login_flow(driver, self.api_key, self.redirect_uri, self.token_path)
    
    def get_historical_price(self, ticker, start_date, end_date, period_type, frequency_type, frequency):
        if type(ticker) is list:
            # filter:
            # tickers equal AND
            # start_date <= datetime <= end_date
            res = self.data[(self.data['ticker'].isin(ticker)) & 
                            (self.data['datetime'] <= end_date) & 
                            (self.data['datetime'] >= start_date)].copy()
        else:
            # filter:
            # tickers equal AND
            # start_date <= datetime <= end_date
            res = self.data[(self.data['ticker'] == ticker) & 
                            (self.data['datetime'] <= end_date) & 
                            (self.data['datetime'] >= start_date)].copy()
        
        return res
