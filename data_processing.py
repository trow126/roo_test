import pandas as pd
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

    # ラグ特徴量 (過去の出来高) - 必要であれば追加
    # for period in lag_periods:
    #     df[f'volume_lag_{period}'] = df['volume'].shift(period)
    #     print(f"Added Volume Lag {period}")

    # 特徴量追加によって発生したNaNを削除 (主に期間の最初の方)
    initial_rows = len(df)
    df.dropna(inplace=True)
    removed_rows = initial_rows - len(df)
    print(f"Removed {removed_rows} rows with NaN values after feature engineering.")

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
def run_data_processing_pipeline(filepath, sma_periods=[5, 10, 20, 50], lag_periods=[1, 2, 3, 5]):
    """データ処理パイプライン全体を実行し、分割済みデータを返す"""
    df = load_data(filepath)
    df = preprocess_data(df)
    df = create_target_variable(df)
    df = add_features(df, sma_periods, lag_periods)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    return X_train, y_train, X_val, y_val, X_test, y_test