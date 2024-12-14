from flask import Flask, request, jsonify
from flask_cors import CORS  # 引入 CORS
import pandas as pd
import numpy as np
import joblib
import os
import logging

# 初始化 Flask 应用
app = Flask(__name__)

# 启用 CORS，允许来自 https://carpark.azurewebsites.net 的请求
CORS(app, origins="https://carpark.azurewebsites.net/", supports_credentials=True)

# 设置日志记录
logging.basicConfig(level=logging.INFO,  # 设置日志级别
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.FileHandler("app.log"),  # 将日志写入文件
                        logging.StreamHandler()  # 同时输出到控制台
                    ])

# 模型目录
MODEL_DIR = "models"
FEATURES_FILE = "features.pkl"

# 载入共用的特征名称
try:
    features_path = os.path.join(MODEL_DIR, FEATURES_FILE)
    feature_names = joblib.load(features_path)
    logging.info(f"[INFO] 特征名称加载成功: {features_path}")
except FileNotFoundError as e:
    logging.error(f"[ERROR] 无法加载特征文件: {e}")
    feature_names = []

def load_model(car_park_id):
    """
    根据停车场 ID 动态加载对应的模型。
    """
    model_path = os.path.join(MODEL_DIR, f"best_random_forest_model_{car_park_id}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    return joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    """
    接收前端传来的 JSON 数据，基于对应停车场模型进行预测。
    """
    if not feature_names:
        logging.error("特征名称未加载，无法进行预测。")
        return jsonify({"error": "特征名称未加载，无法进行预测。"}), 500

    try:
        # 获取前端发送的 JSON 数据
        data = request.json
        if not data:
            logging.warning("未提供数据")
            return jsonify({"error": "未提供数据"}), 400

        # 验证是否包含 CarParkID
        car_park_id = data.get("CarParkID")
        if not car_park_id:
            logging.warning("未提供 CarParkID")
            return jsonify({"error": "请提供 CarParkID"}), 400

        # 加载对应的模型
        try:
            model = load_model(car_park_id)
        except FileNotFoundError as e:
            logging.error(f"未找到模型文件: {str(e)}")
            return jsonify({"error": str(e)}), 400

        # 构建特征 DataFrame
        features = pd.DataFrame([data])
        features = features.drop(columns=["CarParkID"])  # 移除 CarParkID

        # 检查是否包含所有训练模型时的特征
        missing_features = [f for f in feature_names if f not in features.columns]
        extra_features = [f for f in features.columns if f not in feature_names]

        if missing_features:
            logging.warning(f"缺少特征: {missing_features}")
        if extra_features:
            logging.warning(f"额外的特征: {extra_features}")

        # 填补缺失特征为 0，移除多余的特征
        for feature in missing_features:
            features[feature] = 0
        features = features[feature_names]

        # 处理缺失值或无效值
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 使用模型进行预测
        predictions = model.predict(features)

        # 返回预测结果
        logging.info(f"预测成功: {predictions[0]}")
        return jsonify({"prediction": predictions[0]})
    except Exception as e:
        logging.error(f"预测失败: {str(e)}")
        return jsonify({"error": f"预测失败：{str(e)}"}), 400

if __name__ == '__main__':
    # 启动 Flask 应用
    port = int(os.environ.get("PORT", 8080))  # 确保绑定到 PORT 环境变量
    logging.info(f"Flask 应用启动，绑定端口 {port}")
    app.run(host="0.0.0.0", port=port)
