from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask_cors import CORS
import pymysql

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "novel_db",
    "charset": "utf8mb4"
}

# 模型配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./sentiment_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

ID2EMO = {0: "中性", 1: "吐槽", 2: "虐点", 3: "爽点"}
EMO2TAG = {"中性": "中性文", "吐槽": "吐槽文", "虐点": "虐文", "爽点": "爽文"}

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return ID2EMO[torch.argmax(outputs.logits, dim=1).item()]

def get_books_by_tag(tag):
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("SELECT title, author, tag FROM novel WHERE tag = %s", (tag,))
        books = cur.fetchall()
        cur.close()
        conn.close()
        return books
    except Exception as e:
        print(e)
        return []

# 测试根路由，不404
@app.route('/')
def home():
    return "Flask后端运行正常，前端请启动Vue项目"

@app.route("/api/analyze", methods=["POST"])
def analyze():
    text = request.json.get("text", "")
    emotion = predict_emotion(text)
    tag = EMO2TAG[emotion]
    books = get_books_by_tag(tag)
    return jsonify({"emotion": emotion, "books": books})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
