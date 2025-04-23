from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import joblib # モデル保存/ロードのために追加

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
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='logloss',  # 評価指標
                  callbacks=[lgb.early_stopping(10, verbose=False)])  # 10ラウンド改善なければ停止
    else:
        model.fit(X_train, y_train)
    print(f"{type(model).__name__} training finished.")
    return model

# --- モデル評価 ---
def evaluate_model(model, X_val, y_val, model_name="Model"):
    """モデルを評価する"""
    print(f"\n--- Evaluating {model_name} on Validation Set ---")
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    return val_accuracy

# --- モデル学習と評価をまとめて行う ---
def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """複数のモデルを学習し、評価する"""
    print("Training and evaluating models...")

    models = {
        "Logistic Regression": LogisticRegression(solver=MODEL_LR_SOLVER, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=MODEL_RF_N_ESTIMATORS, random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(random_state=MODEL_LGBM_RANDOM_STATE)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n--- Starting {name} ---")
        # モデル学習
        trained_model = train_model(model, X_train, y_train, X_val, y_val)

        # モデル評価
        test_accuracy = evaluate_model(trained_model, X_test, y_test, name)
        results[name] = test_accuracy
        trained_models[name] = trained_model

        # 精度目標 (55%) の確認
        if test_accuracy >= 0.55:
            print(f"\nAccuracy target (>= 55%) achieved for {name}!")
        else:
            print(f"\nAccuracy target (>= 55%) NOT achieved for {name}.")

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