from flask import Flask, request, jsonify
from flask_cors import CORS  # 引入 CORS
import pandas as pd
import numpy as np
import joblib
import os
import logging

# 初始化 Flask 應用
app = Flask(__name__)

# 啟用 CORS，允許來自 https://carpark.azurewebsites.net 的請求
CORS(app, resources={r"/predict": {"origins": "https://carpark.azurewebsites.net"}})

# 設置日誌記錄
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.FileHandler("app.log"),  # 將日誌寫入文件
                        logging.StreamHandler()  # 同時輸出到控制台
                    ])

# 模型目錄
MODEL_DIR = "models"
FEATURES_FILE = "features.pkl"

# 加載共用的特徵名稱
def load_feature_names():
    """加載特徵名稱"""
    try:
        features_path = os.path.join(MODEL_DIR, FEATURES_FILE)
        feature_names = joblib.load(features_path)
        logging.info(f"[INFO] 特徵名稱加載成功: {features_path}")
        return feature_names
    except FileNotFoundError as e:
        logging.error(f"[ERROR] 無法加載特徵文件: {e}")
        return []

feature_names = load_feature_names()

def load_model(car_park_id):
    """根據停車場 ID 動態加載對應的模型"""
    model_path = os.path.join(MODEL_DIR, f"best_random_forest_model_{car_park_id}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    return joblib.load(model_path)

def process_features(data):
    """處理前端傳來的數據，確保特徵完整"""
    features = pd.DataFrame([data])
    features = features.drop(columns=["CarParkID"], errors='ignore')  # 移除 CarParkID

    # 檢查是否包含所有訓練模型時的特徵
    missing_features = [f for f in feature_names if f not in features.columns]
    extra_features = [f for f in features.columns if f not in feature_names]

    if missing_features:
        logging.warning(f"缺少特徵: {missing_features}")
    if extra_features:
        logging.warning(f"額外的特徵: {extra_features}")

    # 填補缺失特徵為 0，移除多餘的特徵
    for feature in missing_features:
        features[feature] = 0
    features = features[feature_names]

    # 處理缺失值或無效值
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return features

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """接收前端傳來的 JSON 數據，基於對應停車場模型進行預測"""
    if request.method == 'OPTIONS':
        # 處理預檢請求
        return jsonify({"message": "CORS 預檢成功"}), 200

    if not feature_names:
        logging.error("特徵名稱未加載，無法進行預測。")
        return jsonify({"error": "特徵名稱未加載，無法進行預測。"}), 500

    try:
        # 獲取前端發送的 JSON 數據
        data = request.json
        if not data:
            logging.warning("未提供數據")
            return jsonify({"error": "未提供數據"}), 400

        # 驗證是否包含 CarParkID
        car_park_id = data.get("CarParkID")
        if not car_park_id:
            logging.warning("未提供 CarParkID")
            return jsonify({"error": "請提供 CarParkID"}), 400

        # 加載對應的模型
        try:
            model = load_model(car_park_id)
        except FileNotFoundError as e:
            logging.error(f"未找到模型文件: {str(e)}")
            return jsonify({"error": str(e)}), 400

        # 處理數據並進行預測
        features = process_features(data)
        predictions = model.predict(features)

        # 返回預測結果
        logging.info(f"預測成功: {predictions[0]}")
        return jsonify({"prediction": predictions[0]})
    except Exception as e:
        logging.error(f"預測失敗: {str(e)}")
        return jsonify({"error": f"預測失敗：{str(e)}"}), 400


@app.route('/predict', methods=['OPTIONS'])
def predict_options():
    """
    處理預檢請求
    """
    response = jsonify({"message": "CORS 預檢成功"})
    response.headers.add("Access-Control-Allow-Origin", "https://carpark.azurewebsites.net")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    return response, 200


if __name__ == '__main__':
    # 啟動 Flask 應用
    port = int(os.environ.get("PORT", 8080))  # 確保綁定到 PORT 環境變量
    logging.info(f"Flask 應用啟動，綁定端口 {port}")
    app.run(host="0.0.0.0", port=port)
