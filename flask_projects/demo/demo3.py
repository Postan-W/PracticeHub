"""
Date: 2022/03/12
"""
#flask第三方表单演示
from flask import Flask, render_template, request, redirect, url_for,make_response,json,jsonify,abort
from wtforms import StringField,PasswordField,SubmitField
from wtforms.validators import DataRequired,EqualTo
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False
# app.config['SECRET_KEY'] = "MINGZHU"
class Config(object):
    DEBUG = True
    JSON_AS_ASCII = False
    SECRET_KEY = "MINGZHU"
    #pip install pymysql，然后头写成如下形式。头写成mysql://的需要mysqlclient库(python2的叫mysqldb库)，但是容易安装失败
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:@127.0.0.1:3306/forflask"#冒号后面是密码，这里没有密码,最后是库名
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True#省略显式提交操作
app.config.from_object(Config)

db = SQLAlchemy(app)
#下面随意创建两个表示例
# class Role(db.Model):
#     __tablename__ = 'role'
#     id = db.Column(db.Integer,primary_key=True)
#     name = db.Column(db.String(32),unique=True)

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(32), unique=True)
    password = db.Column(db.String(50))
    #一个用户可以有多个角色
    # role_id = db.Column(db.Integer,db.ForeignKey('role.id'))


#表单类
class Register(FlaskForm):
    username = StringField(label="用户名",validators=[DataRequired("用户名不能为空")])
    password = PasswordField(label="密码",validators=[DataRequired("密码不能为空")])
    password2 = PasswordField(label="确认密码",validators=[DataRequired("密码不能为空"),EqualTo("password")])
    submit = SubmitField(label="提交")

#增删改查下面演示增加一行数据
@app.route("/register",methods=["GET","POST"])#表单submit请求的还是本接口，所以如果无法处理post则会报错
def register():
    form = Register()
    if request.method == "GET":
        return render_template("reigster.html", form=form)
    elif request.method == "POST":
        if form.validate_on_submit():#表单第一行加上{{form.hidden_tag()}}，否则这个函数总是返回false
            user = User(name=form.username.data,password=form.password.data)
            #加入任务
            db.session.add(user)
            #提交任务
            # db.session.commit()#已配置
            return "注册成功"
        else:
            password = form.password.data
            password2 = form.password2.data
            print(password, password2)
            print(form.validate_on_submit())
            return "密码不一致"

@app.route("/createtable")
def create_usertable():
    with app.app_context():
        db.create_all()
    return "首次创建user表成功"

if __name__ == '__main__':
    # db.drop_all()#清除所有表
    # db.create_all()#创建所有表
    app.run(host="0.0.0.0",port=5000)