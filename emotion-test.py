import pandas as pd
import requests
import json
import time
import os

# ===================== 你的配置（已100%适配） =====================
INPUT_FILE = "./labeled_data.xls"
OUTPUT_FILE = "./AI修正标签_最终版.xlsx"
COMMENT_COL = "content"
LABEL_COL = "label_text"

# 通义千问兼容接口（不会报 choices 错误）
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL_NAME = "qwen-turbo"

# 读取环境变量 KEY
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise Exception("请先运行：$env:DASHSCOPE_API_KEY=\"sk-dd978ee68dab458ea1d6b8ec8e7a1783\"")
# ====================================================================

def ai_judge(comment):
    if not isinstance(comment, str):
        return "中性"

    prompt = f"""
你只能输出以下四个标签之一：爽点、吐槽、虐点、中性

规则：
爽点 = 夸赞、推荐、喜欢、精彩、好看、好评
吐槽 = 不满、油腻、晦气、无聊、不适、错误、批评
虐点 = 哭、虐心、难过、催泪
中性 = 客观、无明显情绪、褒贬都有

只输出一个词！
评论：{comment}
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        result = response.json()
        label = result["choices"][0]["message"]["content"].strip()
        return label if label in ["爽点", "吐槽", "虐点", "中性"] else "中性"
    except Exception as e:
        print(f"AI调用异常：{str(e)[:60]}")
        return "中性"

def main():
    print("=" * 60)
    print("  通义千问 AI 自动标签修正（最终修复版）")
    print("=" * 60)

    # ✅ 修复 xlrd 报错
    try:
        df = pd.read_excel(INPUT_FILE, engine="xlrd")
    except:
        df = pd.read_excel(INPUT_FILE, engine="openpyxl")

    df["原标签"] = df[LABEL_COL]
    labels = []
    print(f"共 {len(df)} 条评论，开始AI识别...\n")

    for i, text in enumerate(df[COMMENT_COL]):
        label = ai_judge(text)
        labels.append(label)
        print(f"第 {i+1} 条 → {label} | {str(text)[:45]}...")
        time.sleep(0.4)

    df["AI修正标签"] = labels
    df[LABEL_COL] = labels
    df.to_excel(OUTPUT_FILE, index=False)

    changed = (df["原标签"] != df["AI修正标签"]).sum()
    print("\n" + "=" * 60)
    print(f"✅ 处理完成！修改了 {changed} 条标签")
    print(f"✅ 文件已保存：{OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()