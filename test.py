from flask import Flask
app = Flask(__name__)

@app.route("/hello",methods=["GET"])
def hello_python():
    return "后台运行中"

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5001,debug=True)