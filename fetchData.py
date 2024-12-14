from datetime import datetime
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# 设置自定义临时目录
custom_temp_folder = os.path.join("C:\\temp_joblib")
os.makedirs(custom_temp_folder, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = custom_temp_folder

print("Joblib 的临时目录设置为:", os.environ["JOBLIB_TEMP_FOLDER"])

# 连接到 MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['ParkingProject']
collection = db['processed_parking_data']

# 设置时间范围
start_date = datetime(2024, 11, 22, 6, 0, 0)
end_date = datetime(2024, 12, 6, 23, 59, 59)

# 获取所有停车场 ID（从数据库中提取）
car_park_ids = collection.distinct("CarParkID")
print(f"找到的停车场 ID: {car_park_ids}")

# 模型和特征存储路径
model_dir = "predict"
os.makedirs(model_dir, exist_ok=True)

# 初始化结果保存列表
results = []

# 遍历每个停车场进行训练
for car_park_id in car_park_ids:
    print(f"\n正在处理停车场 ID: {car_park_id}...")
    
    # 查询数据
    query = {
        "CarParkID": car_park_id,
        "DataCollectTime": {
            "$gte": start_date.strftime('%Y-%m-%dT%H:%M:%S%z'),
            "$lte": end_date.strftime('%Y-%m-%dT%H:%M:%S%z')
        }
    }
    mongo_data = list(collection.find(query))
    
    # 转换为 DataFrame
    df = pd.DataFrame(mongo_data)
    
    # 检查数据是否为空
    if df.empty:
        print(f"[WARNING] 停车场 ID: {car_park_id} 的数据为空，跳过...")
        continue
    
    # 时间字段处理
    df["DataCollectTime"] = pd.to_datetime(df["DataCollectTime"], utc=True)
    df["DataCollectTime"] = df["DataCollectTime"].dt.tz_convert('Asia/Taipei')
    
    # 数据处理
    columns_to_drop = ["_id", "Source", "CarParkID", "SpaceType", "NumberOfSpaces"]
    filtered_df = df.drop(columns=columns_to_drop, errors="ignore")
    
    filtered_df["hour"] = filtered_df["DataCollectTime"].dt.hour
    filtered_df["minute"] = filtered_df["DataCollectTime"].dt.minute
    filtered_df["weekday"] = filtered_df["DataCollectTime"].dt.weekday
    filtered_df["is_weekend"] = filtered_df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    filtered_df = filtered_df.drop(columns=["DataCollectTime"])
    
    # 检查是否存在目标列
    if "AvailableSpaces" not in filtered_df.columns:
        print(f"[WARNING] 停车场 ID: {car_park_id} 缺少 AvailableSpaces 数据，跳过...")
        continue
    
    # 分离特征与目标
    X = filtered_df.drop(columns=["AvailableSpaces"])
    y = filtered_df["AvailableSpaces"]
    
    # 数据集拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # 初始化 GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    # 模型训练
    print(f"停车场 ID: {car_park_id} 开始网格搜索...")
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_
    
    # 模型评估
    y_pred = best_rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"停车场 ID: {car_park_id} 最佳参数: {grid_search.best_params_}")
    print(f"停车场 ID: {car_park_id} Mean Absolute Error: {mae}")
    print(f"停车场 ID: {car_park_id} R^2 Score: {r2}")
    
    # 保存模型和特征
    model_path = os.path.join(model_dir, f"models/best_random_forest_model_{car_park_id}.pkl")
    features_path = os.path.join(model_dir, "models/features.pkl")
    
    joblib.dump(best_rf_model, model_path)
    joblib.dump(list(X.columns), features_path)
    
    print(f"停车场 ID: {car_park_id} 的模型已保存至: {model_path}")
    print(f"停车场 ID: {car_park_id} 的特征文件已保存至: {features_path}")
    
    # 记录结果
    results.append({
        "CarParkID": car_park_id,
        "Model": "RandomForestRegressor",
        "Best Parameters": grid_search.best_params_,
        "MAE": mae,
        "R^2 Score": r2
    })

# 保存结果到 Excel
results_df = pd.DataFrame(results)
excel_path = os.path.join(model_dir, "model_results.xlsx")
results_df.to_excel(excel_path, index=False)
print(f"模型结果已保存至: {excel_path}")
