import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

id2label = {0: "中性", 1: "吐槽", 2: "虐点", 3: "爽点"}
label2id = {"中性": 0, "吐槽": 1, "虐点": 2, "爽点": 3}

MODEL_NAME = "./bert-local"
DATA_FILE = "labeled_data.xlsx"
OUTPUT_DIR = "./sentiment_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_excel(DATA_FILE)
df = df.dropna(subset=['content', 'label'])

if df['label'].dtype in ['int64', 'float64']:
    df['label'] = df['label'].map({0: "中性", 1: "吐槽", 2: "虐点", 3: "爽点"})

df['labels'] = df['label'].map(label2id)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=4, id2label=id2label, label2id=label2id
).to(device)

class CommentDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['content']).strip()
        label = self.data.iloc[idx]['labels']
        inputs = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": inputs["input_ids"].flatten(), "attention_mask": inputs["attention_mask"].flatten(), "labels": torch.tensor(label, dtype=torch.long)}

train_dataset = CommentDataset(train_df, tokenizer)
test_dataset = CommentDataset(test_df, tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions), "f1": f1_score(labels, predictions, average='weighted')}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=6, per_device_train_batch_size=8,
    learning_rate=2e-5, save_strategy="epoch", eval_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="accuracy"
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
