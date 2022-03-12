"""
Date: 2022/03/12
"""
from flask import Flask, render_template, request, redirect, url_for,make_response,json,jsonify,abort
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
#在网页中抛出异常
@app.route("/number/<int:number>")
def get_number(number):
    if number == 1:
        return "first"
    elif number == 2:
        return "second"
    else:
        abort(404)#抛出异常

#自定义错误
@app.errorhandler(404)#处理abort的404异常
def handle_404(err):
    return render_template("error404.html")


#接口数据填充HTML
@app.route("/login2/<name>")
def login2_text(name):
    data2 = {"one":1,"two":[1,2,3,4,5]}#html中引用时dict也可以用点访问子元素(猜测背后还是转为了python支持的[]形式)
    return render_template("login2.html",data=name,data2 = data2)#这里的参数data(名字随便取)和html中的{{}}接收的保持一致
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
