import requests
import time
import json  # 导入 json 库
from bs4 import BeautifulSoup
import random

class NovelSpider:
    def __init__(self):
        self.base_url = "https://www.bqg128.cc"
        # 你找到的榜单接口
        self.api_url = "https://apibi.cc/api/sort?sort=top"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        def fetch_page(self, url, is_api=False):
            """通用的页面/接口请求方法"""
            try:
                # 如果是请求API，需要伪装 Referer，否则会被拦截
                if is_api:
                    # 临时添加 Referer 头
                    self.session.headers.update({
                        'Referer': 'https://www.bqg128.cc/',
                        'Origin': 'https://www.bqg128.cc'
                    })
                
                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                # 如果是API，直接返回JSON数据
                if is_api:
                    return response.json()

                # 如果是网页，返回文本
                response.encoding = response.apparent_encoding
                return response.text
            except Exception as e:
                print(f"   [错误] 请求失败: {e}")
                return None

    def parse_rank_page(self):
        """解析榜单页（直接解析JSON）"""
        print(f"正在请求榜单接口: {self.api_url}")
        json_data = self.fetch_page(self.api_url, is_api=True)

        if not json_data:
            print("   [错误] 获取榜单数据失败")
            return []

        novels = []

        # 根据接口返回的实际结构调整解析逻辑
        # 假设接口返回格式为: {"code": 0, "msg": "success", "data": [...]}
        if json_data.get('code') != 0:
            print(f"   [错误] 接口返回错误: {json_data.get('msg')}")
            return []

        data_list = json_data.get('data', [])

        for item in data_list:
            # 从JSON中提取信息
            title = item.get('articlename', item.get('name', '未知书名'))
            author = item.get('author', '未知作者')
            book_id = item.get('id', item.get('articleid', ''))

            # 构建详情页链接
            # 注意：这个网站的详情页通常是 #/book/{id}/ 的形式
            detail_url = f"{self.base_url}/#/book/{book_id}/"

            novels.append({
                'title': title,
                'author': author,  # 榜单接口里通常直接包含作者
                'url': detail_url,
                'book_id': book_id
            })

        print(f"   榜单解析成功，共找到 {len(novels)} 部小说")
        return novels

    def parse_detail_page(self, book_id):
        """解析详情页获取简介（也尝试用API）"""
        # 很多网站详情页也有对应的API
        # 例如: https://apibi.cc/api/book?id=12345
        detail_api = f"https://apibi.cc/api/book?id={book_id}"

        json_data = self.fetch_page(detail_api, is_api=True)

        if json_data and json_data.get('code') == 0:
            # 如果API请求成功，直接提取简介
            intro = json_data['data'].get('intro', '暂无简介')
            # 清理简介中的HTML标签（如果有的话）
            intro = BeautifulSoup(intro, 'html.parser').text.strip()
            return intro

        # 如果API失败，再尝试请求HTML页面（作为备选方案）
        # 注意：对于动态页面，requests获取的HTML可能没有简介
        html_url = f"{self.base_url}/book/{book_id}/"
        html = self.fetch_page(html_url)
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            # 这里保留之前的逻辑，作为备选
            dd_tag = soup.select_one('dd')
            if dd_tag:
                return dd_tag.text.strip()

        return "暂无简介"

    def run(self):
        """主运行逻辑"""
        novels = self.parse_rank_page()

        for i, novel in enumerate(novels):
            print(f"\n--- 正在处理 ({i+1}/{len(novels)}): {novel['title']} ---")

            # 既然榜单接口里已经有了作者，这里可以不用再爬，或者用来验证
            author = novel['author']

            # 获取简介（尝试通过API获取）
            description = self.parse_detail_page(novel['book_id'])

            print(f"   -> 作者: {author}")
            print(f"   -> 简介: {description[:50]}...") # 打印前50字预览

            # 这里可以添加保存到数据库或文件的逻辑
            # save_to_db(novel['title'], author, description)

            time.sleep(random.uniform(1, 3)) # 随机延时，防止被封

if __name__ == "__main__":
    spider = NovelSpider()
    spider.run()