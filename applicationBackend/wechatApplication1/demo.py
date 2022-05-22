"""
Date: 2022/04/30
"""
from flask import Flask
app = Flask(__name__)

@app.route("/welcome")
def demo():
    return "来自测试后端的返回"

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)