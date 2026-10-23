from pathlib import Path  # 读取当前脚本所在的文件夹。
import matplotlib.pyplot as plt  # 用于画准确率曲线和混淆矩阵。
import pandas as pd  # 用于读取表格数据。
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score  # 导入评价指标。
from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集。
from sklearn.pipeline import make_pipeline  # 用于把标准化和逻辑回归连起来。
from sklearn.preprocessing import StandardScaler  # 用于统一各列数据的大小范围。
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型。
from sklearn.tree import DecisionTreeClassifier  # 导入决策树模型。

folder = Path(__file__).parent  # 找到本程序所在的文件夹。
data = pd.read_csv(folder / "E-commerce Customer Behavior.csv")  # 读取真实数据，原 CSV 不会被修改。
excluded_cols = ["Customer ID", "Gender", "City", "Satisfaction Level", "Days Since Last Purchase"]  # 去掉编号、非行为信息和用来生成答案的列。
X = pd.get_dummies(data.drop(columns=excluded_cols), dtype=int)  # 只在内存中筛选特征，并把会员类型转成数字列。
y = (data["Days Since Last Purchase"] <= 30).astype(int)  # 定义：近 30 天购买过为 1，否则为 0。
print("本实验预测的是近 30 天是否购买；排除的列：", excluded_cols)  # 说明目标和被排除的列。
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # 留出 20% 数据作为最终测试集。
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev)  # 其余数据再分成训练集和验证集。

c_values = [0.01, 0.1, 1, 10, 100]  # C 越小，逻辑回归的正则化越强。
lr_train_scores = []  # 保存不同 C 下的训练准确率。
lr_val_scores = []  # 保存不同 C 下的验证准确率。
for c in c_values:  # 依次尝试每个 C。
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))  # 先标准化，再训练逻辑回归。
    model.fit(X_train, y_train)  # 用训练集学习。
    lr_train_scores.append(model.score(X_train, y_train))  # 记录训练准确率。
    lr_val_scores.append(model.score(X_val, y_val))  # 记录验证准确率。

depth_values = [1, 2, 3, 4, 5, 6, 7, 8]  # 尝试八种决策树最大深度。
tree_train_scores = []  # 保存不同深度下的训练准确率。
tree_val_scores = []  # 保存不同深度下的验证准确率。
for depth in depth_values:  # 依次尝试每个深度。
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)  # 创建决策树。
    model.fit(X_train, y_train)  # 用训练集学习。
    tree_train_scores.append(model.score(X_train, y_train))  # 记录训练准确率。
    tree_val_scores.append(model.score(X_val, y_val))  # 记录验证准确率。

best_c = c_values[lr_val_scores.index(max(lr_val_scores))]  # 选择验证准确率最高的 C。
best_depth = depth_values[tree_val_scores.index(max(tree_val_scores))]  # 选择验证准确率最高的深度。
lr_model = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=1000))  # 建立选定参数的逻辑回归。
tree_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)  # 建立选定深度的决策树。

for name, english_name, model in [("逻辑回归", "Logistic Regression", lr_model), ("决策树", "Decision Tree", tree_model)]:  # 分别评价两种模型。
    model.fit(X_dev, y_dev)  # 用训练集和验证集合起来的数据重新训练。
    predictions = model.predict(X_test)  # 预测测试集是否购买。
    print(f"{name}：准确率={accuracy_score(y_test, predictions):.3f}，精确率={precision_score(y_test, predictions, zero_division=0):.3f}，召回率={recall_score(y_test, predictions, zero_division=0):.3f}")  # 打印三个指标。
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=["No", "Yes"], cmap="Blues")  # 绘制混淆矩阵。
    plt.title(f"{english_name} confusion matrix")  # 添加英文图题，避免电脑缺少中文字体。
    plt.tight_layout()  # 调整图中空白。
    plt.savefig(folder / f"{name}_混淆矩阵.png", dpi=150)  # 保存图片。
    plt.close()  # 关闭当前图片。

new_user = pd.DataFrame([{"Age": 30, "Membership Type": "Silver", "Total Spend": 800, "Items Purchased": 10, "Average Rating": 4.0, "Discount Applied": True}])  # 准备一位新用户的数据。
new_user = pd.get_dummies(new_user, dtype=int).reindex(columns=X.columns, fill_value=0)  # 让新用户的列与训练数据保持一致。
print("新用户预测：逻辑回归=", lr_model.predict(new_user)[0], "决策树=", tree_model.predict(new_user)[0])  # 输出预测，1 表示购买。
print("选用参数：逻辑回归 C=", best_c, "决策树最大深度=", best_depth)  # 输出选用的参数。

plt.figure(figsize=(9, 4))  # 创建准确率曲线图片。
plt.subplot(1, 2, 1)  # 左图显示逻辑回归。
plt.plot(c_values, lr_train_scores, marker="o", label="Train")  # 画训练准确率。
plt.plot(c_values, lr_val_scores, marker="o", label="Validation")  # 画验证准确率。
plt.xscale("log")  # 让跨度较大的 C 值更容易看清。
plt.xlabel("C")  # 设置横轴。
plt.ylabel("Accuracy")  # 设置纵轴。
plt.title("Logistic Regression")  # 设置标题。
plt.legend()  # 显示图例。
plt.subplot(1, 2, 2)  # 右图显示决策树。
plt.plot(depth_values, tree_train_scores, marker="o", label="Train")  # 画训练准确率。
plt.plot(depth_values, tree_val_scores, marker="o", label="Validation")  # 画验证准确率。
plt.xlabel("Max depth")  # 设置横轴。
plt.title("Decision Tree")  # 设置标题。
plt.legend()  # 显示图例。
plt.tight_layout()  # 调整图中空白。
plt.savefig(folder / "准确率变化曲线.png", dpi=150)  # 保存准确率变化曲线。
plt.close()  # 关闭图片。
