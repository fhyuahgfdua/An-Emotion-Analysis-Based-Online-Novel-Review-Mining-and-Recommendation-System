from flask import Flask, request, jsonify
from models.db_models import db, User, Novel, Comment  # 导入数据库模型
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ================= 1. Flask 基础配置 =================
app = Flask(__name__)

# 数据库配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/novel_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化数据库
db.init_app(app)

# ================= 2. AI 模型加载 (核心链接点) =================
# 确保这里的路径是你 train_model.py 训练后保存的实际路径
MODEL_PATH = "./sentiment_model/checkpoint-600"

print(f"🚀 正在加载模型: {MODEL_PATH} ...")

# 自动检测是否有 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载分词器和模型
# 注意：加载过程可能需要几秒钟，请耐心等待
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)

# 设置为评估模式 (关闭 Dropout 等训练专用层)
model.eval()
print("✅ 模型加载完成！服务已启动。")

# ================= 3. 预测逻辑函数 =================
def predict_sentiment(text):
    """
    接收文本，返回预测的标签和置信度
    """
    # 1. 分词与编码
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(DEVICE)

    # 2. 推理 (不计算梯度以节省内存)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # 计算概率分布
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # 3. 获取结果
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    confidence_score = probabilities[0][predicted_class_id].item()

    # 4. 映射标签 (确保这里的 ID 映射和你训练时的一致)
    # 假设你的模型 config 中保存了 id2label 映射，如果没有，需要手动定义
    # 例如: id2label = {0: "中性", 1: "爽点", 2: "虐点", 3: "吐槽"}
    label_map = model.config.id2label
    predicted_label = label_map.get(predicted_class_id, str(predicted_class_id))

    return predicted_label, round(confidence_score, 4)

# ================= 4. API 路由定义 =================

@app.route('/')
def index():
    return """
    <h1>小说评论情感分析系统</h1>
    <p>✅ 后端运行正常</p>
    <p>🤖 AI 模型已加载</p>
    <p>👉 使用 POST 请求访问 <code>/api/analyze</code> 进行测试</p>
    """

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    验收标准接口：
    接收 JSON: {"text": "主角逆袭了，太爽了！"}
    返回 JSON: {"label": "爽点", "score": 0.98}
    """
    # 1. 获取请求数据
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "缺少 'text' 字段"}), 400

    text = data['text']

    if not text.strip():
        return jsonify({"error": "文本不能为空"}), 400

    # 2. 调用 AI 预测函数
    try:
        label, score = predict_sentiment(text)

        # 3. 构造返回结果 (符合验收标准格式)
        result = {
            "label": label,
            "score": score
        }

        # (可选) 如果你想同时把这条数据存入数据库，可以在这里添加 db 操作代码
        # new_comment = Comment(content=text, sentiment=label)
        # db.session.add(new_comment)
        # db.session.commit()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= 5. 启动入口 =================
if __name__ == '__main__':
    # debug=True 开启热重载，修改代码后自动重启
    app.run(host='0.0.0.0', port=5000, debug=True)
