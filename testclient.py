import json
import os
import pandas as pd
from enum import Enum

class TestClient:
    class PeriodType(Enum):
            DAY = 'day'
            MONTH = 'month'
            YEAR = 'year'
            YEAR_TO_DATE = 'ytd'

    class FrequencyType(Enum):
        MINUTE = 'minute'
        DAILY = 'daily'
        WEEKLY = 'weekly'
        MONTHLY = 'monthly'

    class Frequency(Enum):
        # Minute
        EVERY_MINUTE = 1
        EVERY_FIVE_MINUTES = 5
        EVERY_TEN_MINUTES = 10
        EVERY_FIFTEEN_MINUTES = 15
        EVERY_THIRTY_MINUTES = 30

        # Other frequencies
        DAILY = 1
        WEEKLY = 1
        MONTHLY = 1

    def __init__(self, data_path='./data'):
        self.data_path = data_path
        self.data = None

    def load(self, universe):
        d = []
        for file in os.listdir(self.data_path):
            if file[-5:] == '.json' and file[:-5] in universe:
                df = pd.DataFrame(json.load(open(self.data_path + '/' + file))['candles'])
                df['ticker'] = file[:-5]
                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                df['datetime'] = df['datetime'].dt.date
                d.append(df)
        self.data = pd.concat(d)
    
    def get_historical_price(self, ticker, start_date, end_date, period_type, frequency_type, frequency):
        # filter:
        # tickers equal AND
        # start_date <= datetime <= end_date
        res = self.data[(self.data['ticker'] == ticker) & 
                        (self.data['datetime'] <= end_date) & 
                        (self.data['datetime'] >= start_date)].copy()
        

        # Not doing aggregates b/c I don't think we need them. Especially b/c I don't feel like doing them
        return res.copy()
