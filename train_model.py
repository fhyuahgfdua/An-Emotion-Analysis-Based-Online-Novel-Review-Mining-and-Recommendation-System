import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# =================配置区域=================
# 1. 关键修改：强制使用本地文件夹
MODEL_NAME = "./bert-local"       # 👈 这里必须是指向你文件夹的路径，不是网址！
DATA_FILE = "labeled_data.xlsx"   # 你的数据文件
OUTPUT_DIR = "./sentiment_model" # 模型保存路径
NUM_LABELS = 4                    # 分类数量
LABEL_LIST = ["中性", "吐槽", "虐点", "爽点"] 
# =========================================

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 正在使用设备: {device}")

# 1. 加载数据
print("📊 正在加载数据...")
try:
    df = pd.read_excel(DATA_FILE)
except FileNotFoundError:
    print(f"❌ 错误：找不到文件 {DATA_FILE}，请先运行 data_preprocess.py 并人工标注！")
    exit()

df = df.dropna(subset=['content', 'label'])
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
df['labels'] = df['label'].map(label2id)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"✅ 训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")

# 2. 加载模型和分词器
# 由于是离线模式，这里会直接去读取 "./bert-local" 文件夹
print("📂 正在加载本地 BERT 模型...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS,
    id2label={i: l for i, l in enumerate(LABEL_LIST)},
    label2id=label2id
).to(device)

# 3. 定义数据集类 (保持不变)
class CommentDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['content'])
        label = self.data.iloc[idx]['labels']
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 4. 准备数据
train_dataset = CommentDataset(train_df, tokenizer)
test_dataset = CommentDataset(test_df, tokenizer)

# 5. 评估指标
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}

# 6. 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# 7. 开始训练
print("🔥 开始训练... (请耐心等待)")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

print(f"💾 训练完成！模型已保存至: {OUTPUT_DIR}")