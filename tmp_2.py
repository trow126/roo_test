import xgboost as xgb
import numpy as np

print("-" * 30)
print(f"Attempting to use XGBoost version: {xgb.__version__}")
print("-" * 30)

# ダミーデータ作成
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)
X_val = np.random.rand(20, 5)
y_val = np.random.randint(0, 2, 20)

# XGBoost DMatrix 形式に変換
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# パラメータ設定 (ネイティブAPI用)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1 # learning_rate
}

# 評価用データセットのリスト
evals = [(dtrain, 'train'), (dval, 'eval')]

num_boost_round = 100 # 最大イテレーション数

try:
    print("Calling xgb.train with early_stopping_rounds...")
    # ネイティブAPIの train 関数を呼び出す
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=10, # ネイティブAPIではこの引数が一般的
        verbose_eval=False # 学習過程のログ抑制
    )
    print("SUCCESS: xgb.train with early_stopping_rounds completed without error!")
except Exception as e:
    print(f"ERROR: An unexpected error occurred during xgb.train: {e}")

print("-" * 30)