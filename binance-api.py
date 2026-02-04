import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta,timezone
import pdb

# funding rate 

start = '2023-02-02 00:00:00'
end = '2026-02-02 00:00:00'

def to_milliseconds(dt_str):
    """Convert 'YYYY-MM-DD HH:MM:SS' to Binance timestamp (ms)"""
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def get_funding_rates(symbol, start_time, end_time, limit=1000):
    BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {
        "symbol": symbol,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def fetch_funding_for_period(symbol, start_str, end_str):
    start_ts = to_milliseconds(start_str)
    end_ts = to_milliseconds(end_str)

    all_data = []
    current_start = start_ts

    while True:
        data = get_funding_rates(
            symbol=symbol,
            start_time=current_start,
            end_time=end_ts,
            limit=1000
        )

        if not data:
            break

        all_data.extend(data)
        last_time = data[-1]["fundingTime"]

        print(f"Fetched up to {pd.to_datetime(last_time, unit='ms')}")

        # Advance the window
        current_start = last_time + 1

        time.sleep(0.2)

        # Stop if response size < limit
        if len(data) < 1000:
            break

    df = pd.DataFrame(all_data)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    df.set_index("fundingTime", inplace=True)
    daily_funding = (
        df
        .assign(date=df.index.date)
        .groupby("date")["fundingRate"]
        .mean()
        .to_frame("funding_rate")
    )

    daily_funding.index = pd.to_datetime(daily_funding.index)
    return daily_funding

def get_fear_and_greed(limit=30):
    url = f"https://api.alternative.me/fng/?limit={limit}"
    response = requests.get(url)
    data = response.json()['data']

    history = []
    if response.status_code == 200 and data:
        for item in data:
            ts = int(item["timestamp"])
            date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            history.append({
                "date": date,
                "value": int(item["value"]),
                "sentiment": item["value_classification"]
            })
    history_df = pd.DataFrame(history)
    history_df.set_index("date", inplace=True)
    return history_df

def get_daily_volumes(symbol, start_time, end_time):
    BASE_URL = "https://api.binance.com/api/v3/klines"
    start_ms = to_milliseconds(start_time)
    end_ms = to_milliseconds(end_time)

    total_base_volume = 0.0
    total_quote_volume = 0.0
    daily_data = []

    while start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000  # max allowed per request
        }

        response = requests.get(BASE_URL, params=params)
        data = response.json()
        if not data:
            break

        for candle in data:
            open_time = candle[0]
            date = datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

            base_volume = float(candle[5])   # amount of base asset traded that day
            quote_volume = float(candle[7])  # value traded in quote currency

            daily_data.append({
                "date": date,
                "base_volume": base_volume,
                "quote_volume": quote_volume
            })
        last_open_time = data[-1][0]
        start_ms = last_open_time + 1

    daily_df = pd.DataFrame(daily_data)
    daily_df.set_index("date", inplace=True)
    daily_df.index = pd.to_datetime(daily_df.index)
    return daily_df

def get_daily_price_vol(symbol, start_time, end_time): # returns price, lag_ret and volatility 
    BASE_URL = "https://api.binance.com/api/v3/klines"
    start_ms = to_milliseconds(start_time)
    end_ms = to_milliseconds(end_time)

    total_base_volume = 0.0
    total_quote_volume = 0.0
    daily_data = []

    while start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000  # max allowed per request
        }

        response = requests.get(BASE_URL, params=params)
        data = response.json()
        if not data:
            break

        for candle in data:
            open_time = candle[0]
            date = datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

            base_volume = float(candle[5])   # amount of base asset traded that day
            quote_volume = float(candle[7])  # value traded in quote currency
            close_price = float(candle[4])

            daily_data.append({
                "date": date,
                "close_price":close_price
            })
        last_open_time = data[-1][0]
        start_ms = last_open_time + 1

    daily_df = pd.DataFrame(daily_data)
    daily_df.set_index("date", inplace=True)
    daily_df.index = pd.to_datetime(daily_df.index)
    daily_df['log_returns'] = np.log(daily_df['close_price']/daily_df['close_price'].shift())
    daily_df['30d_vol']= daily_df['log_returns'].rolling(30).std() 
    return daily_df

btc_volume = get_daily_volumes('BTCUSDT', start,end)
btc_price = get_daily_price_vol('BTCUSDT', '2023-01-01 00:00:00', end)
btc_funding = fetch_funding_for_period('BTCUSDT', start,end)
btc = pd.concat([btc_volume, btc_funding, btc_price], axis =1, join = 'inner')
btc = btc.rename(columns={"base_volume": "volume_in_BTC", "quote_volume":"volume_in_USDT"})

filename = f"BTC-2023-02_to_2026-02.csv"
btc.to_csv(filename, index=True)

doge_volume = get_daily_volumes('DOGEUSDT', start,end)
doge_price = get_daily_price_vol('DOGEUSDT', '2023-01-01 00:00:00', end)
doge_funding = fetch_funding_for_period('DOGEUSDT', start,end)
doge = pd.concat([doge_volume, doge_funding, doge_price], axis =1, join = 'inner')
doge = doge.rename(columns={"base_volume": "volume_in_DOGE", "quote_volume":"volume_in_USDT"})

filename = f"DOGE-2023-02_to_2026-02.csv"
doge.to_csv(filename, index=True)

fear_greed = get_fear_and_greed(limit = 365*2)
filename = f"fear-greed.csv"
fear_greed.to_csv(filename, index=True)
# pdb.set_trace()

# symbol = "BTCUSDT"
# period = "1d"

# def get_binance_oi_history(symbol="BTCUSDT", interval="1d", days=365):
#     base_url = "https://fapi.binance.com/futures/data/openInterestHist"

#     end_time = int(time.time() * 1000)
#     start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

#     all_rows = []

#     while True:
#         params = {
#             "symbol": symbol,
#             "period": interval,
#             "limit": 500,
#             # "startTime": start_time,
#             "endTime": end_time
#         }

#         response = requests.get(base_url, params=params)
#         data = response.json()

#         # Stop if no more pages
#         if not data:
#             break

#         all_rows.extend(data)
#         pdb.set_trace()
#         # Move start_time forward using last row timestamp
#         last_ts = data[-1]["timestamp"]
#         start_time = last_ts + 1

#         print(f"Fetched up to {datetime.utcfromtimestamp(last_ts/1000).date()}")

#         time.sleep(0.2)  # be nice to API

#     df = pd.DataFrame(all_rows)

#     if df.empty:
#         print("No data returned.")
#         return df

#     df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
#     df["open_interest"] = df["sumOpenInterest"].astype(float)
#     df["oi_value_usd"] = df["sumOpenInterestValue"].astype(float)

#     return df[["date", "open_interest", "oi_value_usd"]]

# res = get_binance_oi_history()
# pdb.set_trace()

# def get_binance_oi_history(symbol="BTCUSDT", interval="1d", days=365):
#     url = "https://fapi.binance.com/futures/data/openInterestHist"

#     # start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
#     # now = int(time.time() * 1000)

#     all_rows = []

#     while True:
#         params = {
#             "symbol": "BTCUSDT",
#             "period": "1d",
#             "startTime": 1727740800000,
#             "endTime": 1733011199000,
#             "limit": 500
#         }


#         response = requests.get(url, params=params)
#         data = response.json()

#         # Handle API errors
#         if not isinstance(data, list):
#             print("API Error:", data)
#             break

#         if len(data) == 0:
#             break

#         all_rows.extend(data)

#         last_ts = data[-1]["timestamp"]  # newest row returned
#         print("Fetched up to:", datetime.utcfromtimestamp(last_ts / 1000))

#         # Stop if we reached present
#         if last_ts >= now:
#             break

#         start_time = last_ts + 1
#         time.sleep(0.2)

#     df = pd.DataFrame(all_rows)

#     if df.empty:
#         print("No data returned.")
#         return df

#     df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
#     df["open_interest"] = df["sumOpenInterest"].astype(float)
#     df["oi_value_usd"] = df["sumOpenInterestValue"].astype(float)

#     return df.sort_values("date")[["date", "open_interest", "oi_value_usd"]]

# # res = get_binance_oi_history()


# def find_earliest_oi(symbol="BTCUSDT", interval="1d"):
#     url = "https://fapi.binance.com/futures/data/openInterestHist"
#     now_ms = 1733011199000  # Nov 30, 2024 in ms
#     one_day_ms = 24 * 60 * 60 * 1000

#     for days_back in [30, 60, 90, 120, 150, 180, 240, 300, 365]:
#         start_time = now_ms - days_back * one_day_ms
#         end_time = now_ms

#         params = {
#             "symbol": symbol,
#             "period": interval,
#             "startTime": start_time,
#             "endTime": end_time,
#             "limit": 10
#         }

#         r = requests.get(url, params=params)
#         data = r.json()

#         if isinstance(data, list) and len(data) > 0:
#             print(f"✅ Works {days_back} days back")
#         else:
#             print(f"❌ Fails {days_back} days back:", data)

#         time.sleep(0.2)



# if __name__ == "__main__":
#     symbol = "DOGEUSDT"
#     start = "2025-02-02 00:00:00"
#     end   = "2026-02-02 00:00:00"

#     volumes = get_daily_volumes(symbol, start, end)

#     for day in volumes:
#         print(f"{day['date']} — {day['base_volume']:.2f} {symbol} | {day['quote_volume']:.2f} USDT")

# if __name__ == "__main__":
#     # Change this to the symbol you want
#     symbol = "BTCUSDT"

#     # Your target date range
#     df = fetch_funding_for_period(symbol, "2024-10-01", "2024-11-30")

#     # Save to CSV
#     filename = f"binance_funding_{symbol}_2024-10_to_2024-11.csv"
#     df.to_csv(filename, index=False)
#     print(f"Saved {len(df)} rows to {filename}")

#     print(df.head())
    # hist = get_fear_and_greed(365)

#     pdb.set_trace()
