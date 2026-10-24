from pathlib import Path  # 找到程序所在的文件夹。
import matplotlib.pyplot as plt  # 用于画各阶段的准确率曲线。
import pandas as pd  # 用于读取 CSV 数据。
from sklearn.dummy import DummyClassifier  # 导入只猜多数类的简单基线。
from sklearn.metrics import accuracy_score, precision_score, recall_score  # 导入三个评价指标。
from sklearn.model_selection import train_test_split  # 用于划分数据集。
from sklearn.tree import DecisionTreeClassifier  # 导入决策树模型。

folder = Path(__file__).parent  # 获取当前脚本所在目录。
data = pd.read_csv(folder / "E-commerce Customer Behavior.csv")  # 读取原始 CSV，不修改文件。
X = data[["Age", "Total Spend"]]  # 先只准备年龄和累计消费金额两个特征。
y = (data["Days Since Last Purchase"] <= 30).astype(int)  # 近 30 天购买过记为 1，否则为 0。
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # 留出 20% 作测试集。
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev)  # 其余分成训练集和验证集。

stages = [  # 按顺序列出不优化的基线和三个改进步骤。
    ("Base", ["Age"], DummyClassifier(strategy="most_frequent")),  # 基线：不学习特征，只猜最多的一类。
    ("Age", ["Age"], DecisionTreeClassifier(max_depth=1, random_state=42)),  # 第一步：只用年龄训练一层决策树。
    ("Spend", ["Age", "Total Spend"], DecisionTreeClassifier(max_depth=1, random_state=42)),  # 第二步：加入消费金额，树仍然只有一层。
    ("Depth2", ["Age", "Total Spend"], DecisionTreeClassifier(max_depth=2, random_state=42)),  # 第三步：把决策树最大深度增至两层。
]  # 四个阶段使用相同的数据划分，便于公平比较。
names = []  # 保存阶段名称，用来画图。
val_scores = []  # 保存验证集准确率。
test_scores = []  # 保存测试集准确率。

print("预测目标：近 30 天是否购买；1=是，0=否")  # 说明本数据的标签含义。
for name, columns, model in stages:  # 依次运行四个阶段。
    model.fit(X_train[columns], y_train)  # 先用训练集训练模型。
    val_score = model.score(X_val[columns], y_val)  # 用验证集观察改进效果。
    model.fit(X_dev[columns], y_dev)  # 再把训练集和验证集合起来重新训练。
    pred = model.predict(X_test[columns])  # 最后预测从未参与选模型的测试集。
    test_score = accuracy_score(y_test, pred)  # 计算测试集准确率。
    precision = precision_score(y_test, pred, zero_division=0)  # 计算精确率。
    recall = recall_score(y_test, pred, zero_division=0)  # 计算召回率。
    names.append(name)  # 保存当前阶段名称。
    val_scores.append(val_score)  # 保存当前验证准确率。
    test_scores.append(test_score)  # 保存当前测试准确率。
    print(f"{name}: 验证准确率={val_score:.3f}，测试准确率={test_score:.3f}，精确率={precision:.3f}，召回率={recall:.3f}")  # 打印真实结果。

plt.plot(names, val_scores, marker="o", label="Validation")  # 画验证集准确率。
plt.plot(names, test_scores, marker="o", label="Test")  # 画测试集准确率。
plt.ylim(0, 1)  # 纵轴从 0 到 1，避免放大变化。
plt.ylabel("Accuracy")  # 设置纵轴名称。
plt.title("Step-by-step comparison")  # 设置英文标题，避免中文字体缺失。
plt.legend()  # 显示图例。
plt.tight_layout()  # 调整图中空白。
plt.savefig(folder / "分步优化准确率.png", dpi=150)  # 保存图片，方便写实验报告。
plt.close()  # 关闭图片。
