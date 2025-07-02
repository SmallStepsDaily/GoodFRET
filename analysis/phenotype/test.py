import joblib
model_path = r"model/BF-A549-VIN.pkl"
try:
    data = joblib.load(model_path)
    print("加载成功，内容类型：", type(data))
except EOFError:
    print("文件读取失败，可能文件损坏或不完整")
