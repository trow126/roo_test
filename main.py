# data_processing.py から必要な関数をインポート
from data_processing import run_data_processing_pipeline, TARGET_COLUMN

# model_service.py から必要な関数をインポート
from model_service import train_and_evaluate_models

# --- 定数 ---
DATA_FILE = 'BTCUSDT_5m_data_max.csv'

# --- メイン処理 ---
if __name__ == "__main__":
    # データ処理パイプラインを実行
    X_train, y_train, X_val, y_val, X_test, y_test = run_data_processing_pipeline(DATA_FILE)

    # モデル学習と評価
    model_accuracies, trained_models = train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n--- Model Test Accuracies ---")
    for name, accuracy in model_accuracies.items():
        print(f"{name}: {accuracy:.4f}")

    # 次のステップ: 最適モデルの選択と保存、予測サービスの構築
    # ... (ここにコードを追加) ...

    print("\nScript finished.")
