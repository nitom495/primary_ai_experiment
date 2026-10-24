from pathlib import Path  # 找到程序所在的文件夹。
import pandas as pd  # 读取 CSV 数据。
from sklearn.dummy import DummyClassifier  # 导入只猜多数类的模型。
from sklearn.metrics import accuracy_score, precision_score, recall_score  # 导入评价指标。
from sklearn.model_selection import train_test_split  # 用于划分数据。

folder = Path(__file__).parent  # 获取当前脚本所在目录。
data = pd.read_csv(folder / "E-commerce Customer Behavior.csv")  # 读取原始 CSV，不修改它。
X = data[["Age"]]  # 基线模型不会利用年龄，只是用它保持输入格式一致。
y = (data["Days Since Last Purchase"] <= 30).astype(int)  # 近 30 天购买过为 1，否则为 0。
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # 留出 20% 测试集。
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev)  # 再分训练集和验证集。
model = DummyClassifier(strategy="most_frequent")  # 建立总猜多数类的基线模型。
model.fit(X_train, y_train)  # 用训练集确定哪一类人数最多。
val_score = model.score(X_val, y_val)  # 计算验证集准确率。
model.fit(X_dev, y_dev)  # 用训练集和验证集合起来重新训练。
pred = model.predict(X_test)  # 预测测试集。
print("阶段 0：多数类基线")  # 打印阶段名称。
print(f"验证准确率={val_score:.3f}，测试准确率={accuracy_score(y_test, pred):.3f}")  # 打印两种准确率。
print(f"精确率={precision_score(y_test, pred, zero_division=0):.3f}，召回率={recall_score(y_test, pred, zero_division=0):.3f}")  # 打印另外两个指标。
