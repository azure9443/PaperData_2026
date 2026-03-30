import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取训练数据
train_file = "part.csv"
train_df = pd.read_csv(train_file)
train_texts = train_df["weibo_text"].tolist()
train_labels = train_df["opposite_sex_social_avoidance"].tolist()

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("macbert_pytorch")


# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length',
                                max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }


# 数据拆分
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42)

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# BERT 回归模型
class BertRegressor(nn.Module):
    def __init__(self, bert_model_name="macbert_pytorch"):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.regressor(cls_output).squeeze(-1)


# 初始化模型
model = BertRegressor().to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# 训练循环
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证模型
    model.eval()
    val_loss, val_preds, true_vals = 0, [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_preds.extend(outputs.cpu().numpy())
            true_vals.extend(labels.cpu().numpy())

    # 计算验证指标
    mse = mean_squared_error(true_vals, val_preds)
    rmse = np.sqrt(mse)  # 计算RMSE
    mae = mean_absolute_error(true_vals, val_preds)
    r2 = r2_score(true_vals, val_preds)
    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f} | "
          f"Val Loss: {val_loss / len(val_loader):.4f} | MSE: {mse:.4f}| RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

# 生成训练集的BERT预测结果
model.eval()
train_preds, train_true = [], []
with torch.no_grad():
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask)
        train_preds.extend(outputs.cpu().numpy())
        train_true.extend(labels.cpu().numpy())

# 训练XGBoost修正器
xgb = XGBRegressor()
xgb.fit(np.array(train_preds).reshape(-1, 1), train_true)


# 预测并保存结果到新文件函数
def predict_and_save(input_file, output_file):
    # 读取待预测文件
    df = pd.read_csv(input_file)
    texts = df["text"].tolist()

    # 创建预测数据集
    class PredictionDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            inputs = self.tokenizer(text, truncation=True, padding='max_length',
                                    max_length=self.max_length, return_tensors="pt")
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0)
            }

    # 生成预测
    predict_dataset = PredictionDataset(texts, tokenizer)
    predict_loader = DataLoader(predict_dataset, batch_size=8, shuffle=False)

    model.eval()
    bert_preds = []
    with torch.no_grad():
        for batch in predict_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            bert_preds.extend(outputs.cpu().numpy())

    # XGBoost修正预测
    xgb_preds = xgb.predict(np.array(bert_preds).reshape(-1, 1))

    # 添加预测评分列
    df["预测评分"] = xgb_preds

    # 调整列位置到text列右侧
    text_col_idx = df.columns.get_loc("text")
    df.insert(text_col_idx + 1, "预测评分", df.pop("预测评分"))

    # 保存结果
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存到 {output_file}")


# 使用示例
input_file = "终版.csv"  # 需要预测的文件路径
output_file = "回避预测结果.csv"  # 输出文件路径
predict_and_save(input_file, output_file)