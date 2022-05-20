from reg_exp import Regexp
from word_tokenize import Tokenizer
from log_analyze import LogAnalyzer
import pymysql
from flask import Flask, request, jsonify

app = Flask(__name__)


# 创建Mysql连接
def get_mysql_conn():
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='user', charset='utf8')
    cursor = conn.cursor()
    return cursor


@app.route('/')
def hello_world():
    return 'Hello, World!'


# 用户登录
@app.route('/login', methods=['POST'])
def login():
    params = request.json
    username = params['username']
    pwd = params['pwd']

    cursor = get_mysql_conn()
    try:
        sql = "select * from user where username = '{}' and password = '{}'".format(username, pwd)
        cursor.execute(sql)
        user = cursor.fetchone()
        if user is None:
            return jsonify({
                "success": False,
                "message": "用户不存在"
            })
        else:
            return jsonify({
                "success": True,
                "message": "登录成功"
            })
    except:
        return jsonify({
            "success": False,
            "message": "数据库操作异常, 查询失败"
        })


# 正则匹配服务
@app.route('/reg_exp', methods=['POST'])
def reg_process():
    params = request.json
    birth_data = params['birth_data']
    identity_number = params['identity_number']
    email = params['email']
    percentage = params['percentage']
    share_holder = params['share_holder']

    # 此处重新定义了Regexp的构造函数,仅使用类中方法对post中的字符串数据进行处理
    regexp = Regexp()
    return jsonify({
        "birth_data": regexp.search_birth_data(birth_data),
        "identity_number": regexp.search_identity_number(identity_number),
        "email": regexp.search_email(email),
        "percentage": regexp.search_percentage(percentage),
        "share_holder": regexp.search_share_holder(share_holder)
    })


# 中文分词服务
@app.route('/word_tokenize', methods=['POST'])
def get_wordToken():
    params = request.json
    sentence = params['sentence']

    # 稍微更新了一下Tokenizer的实现
    tokenizer = Tokenizer()
    return jsonify({
        'tokens': tokenizer.word_tokenize(sentence)
    })


# 日志分析服务
@app.route('/log_analyze', methods=['GET'])
def logAnalysis():
    # 这里仅用本地log进行测试, 如果需要采用其他方式可以后续修改
    data_path = '../shannon-bootcamp-data/03_log_analyze/osprey_response.log'
    log_analyzer = LogAnalyzer(data_path)
    return jsonify({
        'most_query': log_analyzer.most_query(),
        'most_query_time': log_analyzer.most_query_time(),
        'most_query_institution': log_analyzer.most_query_institution()
    })


if __name__ == '__main__':
    app.run(debug=True, port=8080)
