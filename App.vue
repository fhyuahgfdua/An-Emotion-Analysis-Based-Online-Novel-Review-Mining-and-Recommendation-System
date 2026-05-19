<template>
  <div class="page-wrap">
    <div class="card">
      <h2 class="title">📖 小说评论情感分析 & 书籍推荐</h2>

      <div class="input-box">
        <textarea
          v-model="text"
          placeholder="请输入小说评论，系统将自动分析情感并推荐对应小说..."
          @keyup.enter="analyze"
        ></textarea>
      </div>

      <div class="btns">
        <button class="analyze-btn" @click="analyze" :disabled="loading">
          {{ loading ? "分析中..." : "开始分析" }}
        </button>
        <button class="clear-btn" @click="clearAll">清空</button>
      </div>

      <div class="result-box" v-if="emotion">
        <div class="emotion-tag">
          情感结果：<span>{{ emotion }}</span>
        </div>

        <div class="book-list">
          <h4>📚 为你推荐的书籍</h4>
          <div class="book-item" v-for="book in books" :key="book.title">
            <div class="book-title">{{ book.title }}</div>
            <div class="book-info">作者：{{ book.author }}｜标签：{{ book.tag }}</div>
          </div>
          <div class="empty" v-if="books.length === 0">暂无匹配书籍</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import axios from "axios";

const text = ref("");
const emotion = ref("");
const books = ref([]);
const loading = ref(false);

// 分析接口
const analyze = async () => {
  if (!text.value.trim()) {
    alert("请输入评论内容");
    return;
  }
  loading.value = true;
  try {
    const res = await axios.post("http://127.0.0.1:5000/api/analyze", {
      text: text.value,
    });
    emotion.value = res.data.emotion;
    books.value = res.data.books;
  } catch (err) {
    alert("后端服务未启动，请先运行 Flask 后端");
    console.error(err);
  } finally {
    loading.value = false;
  }
};

// 清空
const clearAll = () => {
  text.value = "";
  emotion.value = "";
  books.value = [];
};
</script>

<style scoped>
.page-wrap {
  min-height: 100vh;
  background-color: #f5f7fa;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 40px 20px;
  font-family: "Microsoft YaHei", sans-serif;
}

.card {
  width: 640px;
  background: #fff;
  border-radius: 16px;
  padding: 32px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

.title {
  text-align: center;
  color: #2d3748;
  margin: 0 0 24px;
  font-size: 22px;
}

.input-box textarea {
  width: 100%;
  box-sizing: border-box;
  height: 140px;
  padding: 14px 16px;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  font-size: 15px;
  resize: none;
  transition: border 0.2s;
}
.input-box textarea:focus {
  outline: none;
  border-color: #4299e1;
}

.btns {
  display: flex;
  gap: 12px;
  margin: 18px 0;
}
.analyze-btn {
  flex: 1;
  background-color: #4299e1;
  color: white;
  border: none;
  padding: 12px;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  transition: background 0.2s;
}
.analyze-btn:hover {
  background-color: #3182ce;
}
.analyze-btn:disabled {
  background-color: #a0aec0;
  cursor: not-allowed;
}
.clear-btn {
  width: 100px;
  background-color: #edf2f7;
  color: #4a5568;
  border: none;
  border-radius: 8px;
  cursor: pointer;
}
.clear-btn:hover {
  background-color: #e2e8f0;
}

.result-box {
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid #eee;
}
.emotion-tag {
  font-size: 17px;
  margin-bottom: 16px;
}
.emotion-tag span {
  color: #e53e3e;
  font-weight: bold;
  font-size: 18px;
}

.book-list h4 {
  margin: 0 0 12px;
  color: #2d3748;
}
.book-item {
  background: #f7fafc;
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 10px;
}
.book-title {
  font-weight: bold;
  font-size: 16px;
  color: #2d3748;
}
.book-info {
  font-size: 13px;
  color: #718096;
  margin-top: 4px;
}
.empty {
  color: #a0aec0;
  font-size: 14px;
}
</style>