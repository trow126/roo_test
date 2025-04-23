import xgboost as xgb
from xgboost.callback import EarlyStopping
import numpy as np

print("-" * 30)
print(f"Attempting to use XGBoost version: {xgb.__version__}")
print("-" * 30)

X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)
X_val = np.random.rand(20, 5)
y_val = np.random.randint(0, 2, 20)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
early_stopping_callback = EarlyStopping(rounds=5, save_best=True)

try:
    print("Calling model.fit with callbacks...")
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[early_stopping_callback],
              verbose=False)
    print("SUCCESS: model.fit with callbacks completed without TypeError!")
except TypeError as e:
    print(f"ERROR: TypeError occurred during fit: {e}")
    print("This suggests the 'callbacks' argument is not recognized in this XGBoost version/environment.")
except Exception as e:
     print(f"ERROR: An unexpected error occurred: {e}")

print("-" * 30)
# --- 参考: early_stopping_rounds を試す ---
model_esr = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
try:
    print("Calling model.fit with early_stopping_rounds...")
    model_esr.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=5, # こちらの引数を試す
                  verbose=False)
    print("SUCCESS: model.fit with early_stopping_rounds completed without TypeError!")
except TypeError as e:
    print(f"ERROR: TypeError occurred during fit: {e}")
    print("This suggests the 'early_stopping_rounds' argument is not recognized in this XGBoost version/environment.")
except Exception as e:
     print(f"ERROR: An unexpected error occurred: {e}")
print("-" * 30)