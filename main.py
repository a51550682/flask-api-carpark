from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

# 初始化 Flask 應用
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://carpark.azurewebsites.net"}})

# 模型目錄
MODEL_DIR = "models"
FEATURES_FILE = "features.pkl"

# 載入共用的特徵名稱
try:
    features_path = os.path.join(MODEL_DIR, FEATURES_FILE)
    feature_names = joblib.load(features_path)
    print(f"[INFO] 特徵名稱加載成功: {features_path}")
except FileNotFoundError as e:
    print(f"[ERROR] 無法加載特徵文件: {e}")
    feature_names = []

def load_model(car_park_id):
    """
    根據停車場 ID 動態載入對應的模型。
    """
    model_path = os.path.join(MODEL_DIR, f"best_random_forest_model_{car_park_id}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    return joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    """
    接收前端傳來的 JSON 數據，基於對應停車場模型進行預測。
    """
    if not feature_names:
        return jsonify({"error": "特徵名稱未加載，無法進行預測。"}), 500

    try:
        # 獲取前端發送的 JSON 數據
        data = request.json
        if not data:
            return jsonify({"error": "未提供數據"}), 400

        # 驗證是否包含 CarParkID
        car_park_id = data.get("CarParkID")
        if not car_park_id:
            return jsonify({"error": "請提供 CarParkID"}), 400

        # 載入對應的模型
        try:
            model = load_model(car_park_id)
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 400

        # 構建特徵 DataFrame
        features = pd.DataFrame([data])
        features = features.drop(columns=["CarParkID"])  # 移除 CarParkID

        # 檢查是否包含所有訓練模型時的特徵
        missing_features = [f for f in feature_names if f not in features.columns]
        extra_features = [f for f in features.columns if f not in feature_names]

        if missing_features:
            print(f"[WARNING] 缺少特徵: {missing_features}")
        if extra_features:
            print(f"[WARNING] 額外的特徵: {extra_features}")

        # 填補缺失特徵為 0，移除多餘的特徵
        for feature in missing_features:
            features[feature] = 0
        features = features[feature_names]

        # 處理缺失值或無效值
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 使用模型進行預測
        predictions = model.predict(features)

        # 返回預測結果
        return jsonify({"prediction": predictions[0]})
    except Exception as e:
        return jsonify({"error": f"預測失敗：{str(e)}"}), 400

if __name__ == '__main__':
    # 啟動 Flask 應用
    port = int(os.environ.get("PORT", 8080))  # 确保绑定到 PORT 环境变量
    app.run(host="0.0.0.0", port=port)
