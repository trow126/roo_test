from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import joblib # モデル保存/ロードのために追加
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import pandas as pd
import xgboost as xgb
from xgboost.callback import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# data_processing.py から必要な定数をインポート (TARGET_COLUMNなど)
from data_processing import TARGET_COLUMN

# --- 定数 ---
MODEL_LR_SOLVER = 'liblinear'
MODEL_RF_N_ESTIMATORS = 100
MODEL_LGBM_RANDOM_STATE = 42
MODEL_SAVE_DIR = 'models' # モデル保存ディレクトリ

# --- モデル学習 ---
def train_model(model, X_train, y_train, X_val=None, y_val=None):
    """モデルを学習する"""
    print(f"Training {type(model).__name__}...")
    if isinstance(model, lgb.LGBMClassifier) and X_val is not None and y_val is not None:
        # LightGBM (検証データあり)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  # eval_metric='logloss', # 初期化時に指定済みのためfitでは不要なことが多い
                  callbacks=[lgb.early_stopping(10, verbose=False),
                             lgb.log_evaluation(period=0)])
    elif isinstance(model, xgb.XGBClassifier) and X_val is not None and y_val is not None:
        # XGBoost (検証データあり)
        # EarlyStoppingコールバックを作成
        early_stopping_callback = EarlyStopping(
            rounds=10,          # 停止するまでのラウンド数
            save_best=True      # 最良のモデルの状態を保持するかどうか
            # metric_name='logloss' # 必要に応じて監視するメトリック名を指定 (eval_metricで指定したもの)
            # data_name='validation_0'# 必要に応じてeval_setのどのデータを使うか指定 (通常は自動)
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)], # 評価用データ
                  callbacks=[early_stopping_callback], # callbacks引数にリストで渡す
                  verbose=False) # 学習過程ログ抑制
    elif isinstance(model, xgb.XGBClassifier):
        # XGBoost (検証データなし)
        model.fit(X_train, y_train, verbose=False)
    elif isinstance(model, lgb.LGBMClassifier):
        # LightGBM (検証データなし)
        model.fit(X_train, y_train)
    else:
        # その他のモデル
        model.fit(X_train, y_train)
    print(f"{type(model).__name__} training finished.")
    return model

# --- モデル評価 ---
def evaluate_model(model, X, y, data_type="Validation"):
    """モデルを評価する"""
    print(f"\n--- Evaluating {type(model).__name__} on {data_type} Set ---")
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"{data_type} Accuracy: {accuracy:.4f}")
    print(f"{data_type} Classification Report:")
    print(classification_report(y, y_pred))
    print(f"{data_type} Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    # AUCも計算する場合は以下を追加 (roc_auc_scoreをimport)
    # try:
    #     from sklearn.metrics import roc_auc_score
    #     y_prob = model.predict_proba(X)[:, 1]
    #     auc = roc_auc_score(y, y_prob)
    #     print(f"{data_type} AUC: {auc:.4f}")
    # except AttributeError:
    #     print(f"AUC not available for {type(model).__name__}")
    return accuracy

# --- ハイパーパラメータ調整 (LightGBM) ---
def tune_hyperparameters_lgbm(X_train, y_train, X_val, y_val):
    """LightGBMのハイパーパラメータをTimeSeriesSplitを用いて調整する"""
    print("\n--- Tuning LightGBM Hyperparameters ---")

    # 訓練データと検証データを結合 (時系列分割のため)
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    # ハイパーパラメータ探索範囲
    param_dist = {
        'n_estimators': [100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 64, 128],
        'max_depth': [-1, 10, 20],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0],
    }


    # TimeSeriesSplit の設定
    # n_splitsは分割数。test_sizeは各テストセットのサイズ（ここでは検証データサイズに合わせる）
    # gapは訓練セットとテストセットの間の期間（データリーケージ防止）
    tscv = TimeSeriesSplit(n_splits=5, gap=0) # gap=0は簡略化のため。本来は適切に設定

    # RandomizedSearchCV の設定
    # n_iterは試行回数。cvはクロスバリデーションの分割戦略。scoringは評価指標。random_stateで再現性確保。n_jobsで並列処理。
    random_search = RandomizedSearchCV(
        lgb.LGBMClassifier(random_state=MODEL_LGBM_RANDOM_STATE, verbosity=-1),
        param_distributions=param_dist,
        n_iter=50, # 試行回数。計算時間に応じて調整
        scoring='accuracy',
        cv=tscv,
        random_state=MODEL_LGBM_RANDOM_STATE,
        n_jobs=-1, # 利用可能なコアを全て使用
        verbose=0 # 進行状況を表示
    )

    # ハイパーパラメータ調整実行
    random_search.fit(X_combined, y_combined)

    print("Best hyperparameters found:")
    print(random_search.best_params_)
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")

    # 最適なハイパーパラメータでモデルを再学習 (訓練データ+検証データ全体で)
    print("\nTraining LightGBM with best hyperparameters on combined train+validation data...")
    best_model = lgb.LGBMClassifier(**random_search.best_params_, random_state=MODEL_LGBM_RANDOM_STATE, verbosity=-1)
    best_model.fit(X_combined, y_combined, callbacks=[lgb.log_evaluation(period=0)]) # 評価ログ抑制
    print("LightGBM training with best hyperparameters finished.")

    return best_model

# --- モデル学習と評価をまとめて行う ---
def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """複数のモデルを学習し、評価する"""
    print("Training and evaluating models...")

    models = {
        "Logistic Regression": LogisticRegression(solver=MODEL_LR_SOLVER, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=MODEL_RF_N_ESTIMATORS, random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(random_state=MODEL_LGBM_RANDOM_STATE, verbosity=-1),
        "XGBoost": xgb.XGBClassifier(random_state=MODEL_LGBM_RANDOM_STATE, use_label_encoder=False, eval_metric='logloss',verbosity=0)
    }

    results = {}
    trained_models = {}

    # `train_model` の呼び出しロジックを修正 (XGBoostも検証データを使うように)
    for name, model in models.items():
        print(f"\n--- Starting {name} ---")
        if name == "LightGBM":
             # ハイパーパラメータ調整 (関数内でログ制御)
             trained_model = tune_hyperparameters_lgbm(X_train, y_train, X_val, y_val)
        elif name == "XGBoost":
             # XGBoostも検証データを渡して学習 (train_model内でログ制御)
             trained_model = train_model(model, X_train, y_train, X_val, y_val)
        elif isinstance(model, lgb.LGBMClassifier): # チューニングしないLightGBMの場合
             # 検証データを渡して学習 (train_model内でログ制御)
             trained_model = train_model(model, X_train, y_train, X_val, y_val)
        else: # Logistic Regression, Random Forest
             trained_model = train_model(model, X_train, y_train) # 検証データなし

        # 検証データでの評価 (LightGBMの早期停止確認用など)
        if X_val is not None and y_val is not None:
             evaluate_model(trained_model, X_val, y_val, f"{name} Validation")

        # テストデータでの最終評価
        test_accuracy = evaluate_model(trained_model, X_test, y_test, f"{name} Test")
        results[name] = test_accuracy
        trained_models[name] = trained_model

        # 精度目標 (55%) の確認
        if test_accuracy >= 0.55:
            print(f"\nAccuracy target (>= 55%) achieved for {name}!")
        else:
            print(f"\nAccuracy target (>= 55%) NOT achieved for {name}.")

    # --- ベースラインモデルの評価 ---
    print("\n--- Evaluating Baseline Models ---")
    baseline_predictions_test = generate_baseline_predictions(X_test, y_test)

    baseline_results = {}
    for col in baseline_predictions_test.columns:
        baseline_name = col
        baseline_y_pred = baseline_predictions_test[col]
        baseline_accuracy = accuracy_score(y_test, baseline_y_pred)
        baseline_results[baseline_name] = baseline_accuracy

        print(f"\n--- Evaluating {baseline_name} on Test Set ---")
        print(f"Test Accuracy: {baseline_accuracy:.4f}")
        print(f"Test Classification Report:\n{classification_report(y_test, baseline_y_pred)}")
        print(f"Test Confusion Matrix:\n{confusion_matrix(y_test, baseline_y_pred)}")

    # ベースライン結果をresultsディクショナリに追加
    results.update(baseline_results)

    return results, trained_models

# --- モデルの保存 ---
def save_model(model, filepath):
    """学習済みモデルをファイルに保存する"""
    print(f"Saving model to {filepath}...")
    joblib.dump(model, filepath)
    print("Model saved.")

# --- モデルのロード ---
def load_model(filepath):
    """ファイルから学習済みモデルをロードする"""
    print(f"Loading model from {filepath}...")
    model = joblib.load(filepath)
    print("Model loaded.")
    return model

# --- 予測の実行 ---
def predict_with_model(model, X):
    """学習済みモデルを使って予測を実行する"""
    print("Making predictions...")
    predictions = model.predict(X)
    print("Predictions finished.")
    return predictions

# --- ベースライン予測の生成 ---
def generate_baseline_predictions(X, y):
    """ベースラインモデルの予測を生成する"""
    print("\n--- Generating Baseline Predictions ---")

    # ベースライン1: 常にHIGH (1) と予測
    baseline_high_pred = pd.Series(1, index=y.index, name="Baseline_AlwaysHIGH")
    print("Generated Baseline: Always HIGH")

    # ベースライン2: 前の足と同じ方向を予測 (closeの差分を使用)
    # 最初の行はNaNになるため、HIGH(1)と予測
    baseline_prev_direction_pred = (X['close'].diff().shift(1) > 0).astype(int)
    baseline_prev_direction_pred.iloc[0] = 1 # 最初の行を1で埋める
    baseline_prev_direction_pred.name = "Baseline_PrevDirection"
    print("Generated Baseline: Previous Direction")

    # ベースライン3: ランダム予測 (参考用)
    # import numpy as np
    # baseline_random_pred = pd.Series(np.random.randint(0, 2, size=len(y)), index=y.index, name="Baseline_Random")
    # print("Generated Baseline: Random")

    # 複数のベースライン予測をDataFrameにまとめる
    baseline_predictions = pd.DataFrame({
        baseline_high_pred.name: baseline_high_pred,
        baseline_prev_direction_pred.name: baseline_prev_direction_pred,
        # baseline_random_pred.name: baseline_random_pred, # ランダム予測を含める場合
    })

    return baseline_predictions

# --- ipynbでのテスト用エントリポイント (オプション) ---
# if __name__ == '__main__':
#     # このファイルを直接実行した場合のテストコード
#     # ダミーデータを作成するか、data_processingモジュールからデータをロードしてテスト
#     print("Running model_service.py as a script (for testing)...")
#     # 例: ダミーデータでのテスト
#     # import numpy as np
#     # X_train_dummy = np.random.rand(100, 10)
#     # y_train_dummy = np.random.randint(0, 2, 100)
#     # X_val_dummy = np.random.rand(20, 10)
#     # y_val_dummy = np.random.randint(0, 2, 20)
#     # X_test_dummy = np.random.rand(20, 10)
#     # y_test_dummy = np.random.randint(0, 2, 20)
#     #
#     # results, trained_models = train_and_evaluate_model(X_train_dummy, y_train_dummy, X_val_dummy, y_val_dummy, X_test_dummy, y_test_dummy)
#     # print("\nTest Model Accuracies:", results)
#     #
#     # # モデルの保存とロードのテスト
#     # if trained_models:
#     #     sample_model_name = list(trained_models.keys())[0]
#     #     sample_model = trained_models[sample_model_name]
#     #     save_model(sample_model, f"{sample_model_name.replace(' ', '_').lower()}_test_model.pkl")
#     #     loaded_model = load_model(f"{sample_model_name.replace(' ', '_').lower()}_test_model.pkl")
#     #     print(f"Loaded model type: {type(loaded_model)}")
#     #
#     # # 予測のテスト
#     # if loaded_model:
#     #     dummy_prediction_data = np.random.rand(5, 10)
#     #     predictions = predict_with_model(loaded_model, dummy_prediction_data)
#     #     print("Sample predictions:", predictions)