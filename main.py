# data_processing.py から必要な関数をインポート
from data_processing import load_data, preprocess_data, create_target_variable, add_features, split_data, TARGET_COLUMN

# data_processing.py から必要な関数をインポート
from data_processing import load_data, preprocess_data, create_target_variable, add_features, split_data, TARGET_COLUMN

# model_service.py から必要な関数とモデルクラスをインポート
from model_service import train_and_evaluate_models, save_model, load_model, MODEL_SAVE_DIR # train_and_evaluate_modelsを追加
from sklearn.linear_model import LogisticRegression # train_and_evaluate_models内で定義するため不要になるが、import自体は残しておく
from sklearn.ensemble import RandomForestClassifier # train_and_evaluate_models内で定義するため不要になるが、import自体は残しておく
import lightgbm as lgb # train_and_evaluate_models内で定義するため不要になるが、import自体は残しておく
import os

# --- 定数 ---
DATA_FILE = 'BTCUSDT_5m_data_max.csv'

# --- メイン処理 ---
if __name__ == "__main__":
    print("--- データ処理フェーズ ---")
    # データ処理の各ステップを実行
    df = load_data(DATA_FILE)
    df = preprocess_data(df)
    df = create_target_variable(df)
    # add_features内でNaN処理を行うため、ここでは不要
    df = add_features(df) # デフォルトのSMA/Lag期間を使用

    # データ分割
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    print("\n--- モデル構築・評価フェーズ ---")

    # モデル学習と評価をまとめて実行
    results, trained_models = train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n--- Model Test Accuracies ---")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.4f}")

    # オプション: 最適モデルの保存
    # if trained_models:
    #     best_model_name = max(results, key=results.get)
    #     best_model = trained_models[best_model_name]
    #     model_save_path = os.path.join(MODEL_SAVE_DIR, f"{best_model_name.replace(' ', '_').lower()}_best_model.pkl")
    #     os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    #     save_model(best_model, model_save_path)
    #     print(f"\nBest model ({best_model_name}) saved to {model_save_path}")

    print("\n--- Script finished ---")
