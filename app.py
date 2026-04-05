from flask import Flask
from models.db_models import db, User, Novel, Comment # 导入模型

app = Flask(__name__)

# 数据库配置：用户名:密码@localhost/数据库名
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/novel_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化数据库对象
db.init_app(app)

@app.route('/')
def hello():
    return "后端环境搭建成功！数据库已连接。"

# 用于测试建表的临时路由
@app.route('/init-db')
def init_db():
    with app.app_context():
        db.drop_all() # 删除所有表（慎用，这里是为了重置）
        db.create_all() # 创建所有表
        return "数据库表创建成功！请检查 MySQL。"

if __name__ == '__main__':
    app.run(debug=True)