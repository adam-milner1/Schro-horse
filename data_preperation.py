import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Union
import numpy as np


def download_stock_data(tickers, years=5):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def add_OC_CO_next_changes(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Adds two predictive target features for a given ticker:
    - OC_next: Next day's Open to Close change.
    - CO_next: Today's Close to Next day's Open change.
    """

    open_ = data["Open"][ticker]
    close_ = data["Close"][ticker]

    # Next day Open / Close
    open_next = open_.shift(-1)
    close_next = close_.shift(-1)

    # OC_next: next day's Open -> Close change
    data[("OC_next", ticker)] = close_next - open_next

    # CO_next: today's Close -> next day's Open
    data[("CO_next", ticker)] = open_next - close_

    return data


def add_sma(data: pd.DataFrame, ticker: str, window: int) -> pd.DataFrame:
    """Add Simple Moving Average (SMA) for a given ticker and window."""
    data[(f"SMA_{window}", ticker)] = (
        data["Close"][ticker].rolling(window=window).mean()
    )
    return data

def add_ema(data: pd.DataFrame, ticker: str, window: int) -> pd.DataFrame:
    """Add Exponential Moving Average (EMA) for a given ticker and window."""
    data[(f"EMA_{window}", ticker)] = (
        data["Close"][ticker].ewm(span=window, adjust=False).mean()
    )
    return data

def add_macd(data: pd.DataFrame, ticker: str, short=12, long=26, signal=9) -> pd.DataFrame:
    """Add MACD and Signal line for a given ticker."""
    ema_short = data["Close"][ticker].ewm(span=short, adjust=False).mean()
    ema_long = data["Close"][ticker].ewm(span=long, adjust=False).mean()
    
    data[("MACD", ticker)] = ema_short - ema_long
    data[("Signal_Line", ticker)] = data[("MACD", ticker)].ewm(span=signal, adjust=False).mean()
    return data

def add_rsi(data: pd.DataFrame, ticker: str, window: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index (RSI) for a given ticker."""
    close = data["Close"][ticker]
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    data[(f"RSI_{window}", ticker)] = rsi
    return data

def add_atr(data: pd.DataFrame, ticker: str, window: int = 14) -> None:
    """Add Average True Range (ATR) for a given ticker."""
    high = data["High"][ticker]
    low = data["Low"][ticker]
    close = data["Close"][ticker]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    data[(f"ATR_{window}",ticker)] = tr.rolling(window=window).mean()
    return data

def add_change(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add absolute change between Close and Open for a given ticker."""
    data[("Change", ticker)] = data["Close"][ticker] - data["Open"][ticker]
    return data

def add_change_pct(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add percentage change between Close and Open for a given ticker."""
    change = data["Close"][ticker] - data["Open"][ticker]
    denom = data["Close"][ticker] + data["Open"][ticker]
    data[("Change %", ticker)] = 200 * change / denom
    return data


def add_all_indicators(data: pd.DataFrame, tickers: list) -> pd.DataFrame:
    for t in tickers:
        # Short/long SMAs
        data = add_sma(data, t, 20)
        data = add_sma(data, t, 50)
        # Short/long EMAs
        data = add_ema(data, t, 20)
        data = add_ema(data, t, 50)
        # MACD
        data = add_macd(data, t)
        # RSI
        data = add_rsi(data, t, 14)
        # ATR
        data = add_atr(data, t, 14)
        # Change
        data = add_change(data, t)
        # Change %
        data = add_change_pct(data, t)
        # Sort columns
    data = data.sort_index(axis=1, level=0)
    return data

def normalise_data(
    data: pd.DataFrame,
    min_range: float = -np.pi,
    max_range: float = np.pi
) -> pd.DataFrame:
    """Normalise data to a specified range (default [-π, π]) using Min-Max scaling."""

    min_val = data.min()
    max_val = data.max()
    scaled = (data - min_val) / (max_val - min_val)
    return scaled * (max_range - min_range) + min_range

def remove_na(data: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with any NaN values."""
    return data.dropna()


def split_with_targets(
    df: pd.DataFrame,
    target_cols: Union[str, List[str]],
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/test sets for features (X) and targets (y).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target_cols : str or list of str
        Column(s) to use as target(s).
    test_size : float, default=0.2
        Fraction of data to use for test set.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    X = df.drop(columns=target_cols)
    y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def split_time_series(
    df: pd.DataFrame,
    target_cols: Union[str, List[str]],
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sequentially split a DataFrame into train/test sets for time series data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data sorted by time (oldest → newest).
    target_cols : str or list of str
        Column(s) to use as target(s).
    test_size : float, default=0.2
        Fraction of data to use for test set (taken from the end).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    split_index = int(len(df) * (1 - test_size))

    X = df.drop(columns=target_cols)
    y = df[target_cols]

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def two_qubit_data_tickers(tickers):
    data = download_stock_data(tickers)
    for t in tickers:
        data = add_macd(data, t)


        data = add_OC_CO_next_changes(data, t)

        data = add_ema(data, t, 20)

    
    data.sort_index(axis=1, level=0)
    data = remove_na(data)


    X_train, X_test, y_train, y_test = split_time_series(data, target_cols=["OC_next", "CO_next"])

    
    return X_train, X_test, y_train, y_test


