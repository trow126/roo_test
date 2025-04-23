import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # train_test_splitはここでは使わないが、後で使う可能性のためimportしておく

# --- 定数 ---
TARGET_COLUMN = 'target' # 目的変数の列名 (1: HIGH, 0: LOW)

# --- データ読み込み ---
def load_data(filepath):
    """CSVファイルを読み込む"""
    print(f"Loading data from {filepath}...")
    # CSV読み込み (timestampをパース)
    df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
    print("Data loaded.")
    return df

# --- データ前処理 ---
def preprocess_data(df):
    """データの前処理を行う"""
    print("Preprocessing data...")

    # 不要な列があれば削除 (例: turnover) - 必要に応じて調整
    if 'turnover' in df.columns:
        df = df.drop(columns=['turnover'])
        print("Dropped 'turnover' column.")

    # 欠損値の確認と処理 (前方フィル)
    initial_nan_count = df.isnull().sum().sum()
    if initial_nan_count > 0:
        print(f"Found {initial_nan_count} NaN values. Applying forward fill...")
        df.fillna(method='ffill', inplace=True)
        # ffillでも残る場合は削除 (最初の行など)
        df.dropna(inplace=True)
        print("NaN values handled.")
    else:
        print("No NaN values found.")

    # データ型の確認 (必要であれば数値型に変換)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # coerceでNaNになった場合の処理も追加検討
            df.dropna(subset=[col], inplace=True)

    print("Data preprocessing finished.")
    return df

# --- 目的変数の作成 ---
def create_target_variable(df):
    """次の足の終値が現在より高いか低いかで目的変数を作成"""
    print("Creating target variable...")
    # 次の足の終値を取得
    df['next_close'] = df['close'].shift(-1)

    # 目的変数を作成 (1 if next_close > close else 0)
    df[TARGET_COLUMN] = (df['next_close'] > df['close']).astype(int)

    # 最後の行は next_close が NaN になるため削除
    df.dropna(subset=['next_close'], inplace=True)
    df = df.drop(columns=['next_close']) # 不要になった列を削除

    print("Target variable created.")
    print(f"Target distribution:\n{df[TARGET_COLUMN].value_counts(normalize=True)}")
    return df

# --- 特徴量エンジニアリング ---
def add_features(df, sma_periods=[5, 10, 20, 50], lag_periods=[1, 2, 3, 5]):
    """テクニカル指標やラグ特徴量を追加する"""
    print("Adding features...")

    # 移動平均線 (SMA)
    for period in sma_periods:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        print(f"Added SMA {period}")

    # ラグ特徴量 (過去の終値)
    for period in lag_periods:
        df[f'close_lag_{period}'] = df['close'].shift(period)
        print(f"Added Close Lag {period}")

    # --- テクニカル指標 ---
    print("Adding technical indicators...")
    # EMA
    for period in [5, 10, 20]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        print(f"Added EMA {period}")

    # RSI (手計算 - 簡略版)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=14-1, adjust=False).mean()
    avg_loss = loss.ewm(com=14-1, adjust=False).mean()
    # ゼロ除算を防ぐ
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    print("Added RSI 14")

    # MACD (手計算 - 簡略版)
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = signal
    print("Added MACD and MACD Signal")

    # ボリンジャーバンド (%B, Bandwidth) (手計算 - 簡略版)
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    # ゼロ除算を防ぐ
    denominator = upper_band - lower_band
    df['bb_percent_b'] = (df['close'] - lower_band) / denominator.replace(0, 1e-9) * 100
    df['bb_bandwidth'] = (upper_band - lower_band) / rolling_mean.replace(0, 1e-9) * 100
    print("Added Bollinger Bands (%B, Bandwidth)")

    # 価格変動率 (ROC) (手計算)
    df['roc_1'] = df['close'].pct_change(periods=1) * 100
    print("Added ROC 1")

    # ボラティリティ (標準偏差) (手計算)
    df['volatility_14'] = df['close'].rolling(window=14).std()
    print("Added Volatility 14")

    # ラグ特徴量 (過去の出来高)
    for period in lag_periods:
        df[f'volume_lag_{period}'] = df['volume'].shift(period)
        print(f"Added Volume Lag {period}")

    # --- ローソク足の数値的特徴量 ---
    print("Adding candlestick numerical features...")
    df['true_range'] = df['high'] - df['low']
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    print("Added True Range, Body Size, Upper Shadow, Lower Shadow")

    # --- 追加の出来高関連特徴量 ---
    print("Adding additional volume features...")
    # 出来高の移動平均
    for period in [5, 10, 20]:
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        print(f"Added Volume SMA {period}")

    # 出来高の変動率
    df['volume_change'] = df['volume'].pct_change()
    print("Added Volume Change")

    # --- 価格と指標の組み合わせ特徴量 ---
    print("Adding price and indicator combination features...")
    # 価格とSMAの比率
    for period in [5, 10, 20]:
         df[f'price_sma_ratio_{period}'] = df['close'] / df[f'sma_{period}'].replace(0, 1e-9)
         print(f"Added Price SMA Ratio {period}")

    # RSIの差分
    df['rsi_diff'] = df['rsi_14'].diff()
    print("Added RSI Difference")

    # MACDとSignalの差分 (MACD Line - Signal Line)
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    print("Added MACD Histogram")

    # --- 日中パターン・季節性特徴量 ---
    print("Adding intraday pattern and seasonality features...")
    # 時間帯 (0-23)
    df['hour'] = df.index.hour
    print("Added Hour")

    # 曜日 (0=月, 6=日)
    df['day_of_week'] = df.index.dayofweek
    print("Added Day of Week")

    # --- 出来高と価格の相互作用特徴量 ---
    print("Adding volume and price interaction features...")
    # 出来高加重移動平均 (VWAP) - 簡略版 (累積出来高と累積(価格*出来高)を使用)
    df['cum_volume'] = df['volume'].cumsum()
    df['cum_price_volume'] = (df['close'] * df['volume']).cumsum()
    # ゼロ除算を防ぐ
    df['vwap'] = df['cum_price_volume'] / df['cum_volume'].replace(0, 1e-9)
    df = df.drop(columns=['cum_volume', 'cum_price_volume']) # 中間列を削除
    print("Added VWAP")

    # 出来高と価格トレンドの組み合わせ (例: 価格が上昇している期間の出来高の合計)
    # これは少し複雑になるため、まずは単純なVWAPを追加します。
    # 必要であれば、より複雑な特徴量も検討します。

    # 必要であれば、他の時間関連特徴量も追加検討 (例: 週末フラグ、特定の時間帯フラグなど)
    # 特徴量追加によって発生したNaNを削除 (主に期間の最初の方)
    initial_rows = len(df)
    df.dropna(inplace=True)
    removed_rows = initial_rows - len(df)
    print(f"Removed {removed_rows} rows with NaN values after feature engineering.")

    # 無限大の値を含む行を削除
    initial_rows_inf = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True) # 無限大をNaNに変換
    df.dropna(inplace=True) # NaNになった行を削除
    removed_rows_inf = initial_rows_inf - len(df)
    if removed_rows_inf > 0:
        print(f"Removed {removed_rows_inf} rows with infinity values after feature engineering.")

    # 必要であれば、他の組み合わせ特徴量も追加検討
    print("Features added.")
    return df

# --- データ分割 ---
def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """データを時系列で訓練、検証、テストに分割する"""
    data_len = len(df)
    train_end_index = int(data_len * train_ratio)
    val_end_index = int(data_len * (train_ratio + val_ratio))

    train_data = df.iloc[:train_end_index]
    val_data = df.iloc[train_end_index:val_end_index]
    test_data = df.iloc[val_end_index:]

    print("Data split into train, validation, and test sets.")
    print(f"Train set size: {len(train_data)} ({len(train_data)/data_len:.1%})")
    print(f"Validation set size: {len(val_data)} ({len(val_data)/data_len:.1%})")
    print(f"Test set size: {len(test_data)} ({len(test_data)/data_len:.1%})")

    # 特徴量 (X) と目的変数 (y) に分割
    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]
    X_val = val_data.drop(columns=[TARGET_COLUMN])
    y_val = val_data[TARGET_COLUMN]
    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]

    return X_train, y_train, X_val, y_val, X_test, y_test

# --- データ処理パイプライン実行関数 ---
# def run_data_processing_pipeline(filepath, sma_periods=[5, 10, 20, 50], lag_periods=[1, 2, 3, 5]):
#     """データ処理パイプライン全体を実行し、分割済みデータを返す"""
#     df = load_data(filepath)
#     df = preprocess_data(df)
#     df = create_target_variable(df)
#     df = add_features(df, sma_periods, lag_periods)
#     X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
#     return X_train, y_train, X_val, y_val, X_test, y_test