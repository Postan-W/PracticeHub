# class Saver:
#     def __init__(self,provider):
#         db_settings = {"mysql":{
#             "provider": "mysql",  # 声明数据库种类
#             "host": db_host,  # 数据库主机地址，也可以是域名
#             "port": db_port,  # 端口
#             "database": db_database,  # 数据库名
#             "user": db_user,  # 用户名
#             "password": db_password,  # 密码
#             "charset": "utf8mb4",},  # 字符集
#
#                       "oracle":{
#             "provider":"oracle",
#             "user":db_user,
#             "password":db_password,
#             "dsn":"{}:{}/{}".format(db_host,db_port,db_database)}
#                       }
#         dst_setting = db_settings.get(provider)
#         assert dst_setting, "{provider} is not is not supported!".format_map(vars())
#         self.dst_mysql = Database(**dst_setting)
#
#     @db_session()
#     def datain(self,camera_table_name,):
#         now_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         self.dst_mysql.execute(
#               "INSERT INTO \"{camera_table_name}\" (timestamp,inNum,inTime,passNum,avgDur) VALUES (\"{now_datetime}\",\"{inNum}\",\"{inTime}\",\"{passNum}\",\"{avgDur}\")".format_map(vars()))
#         print("data save ok!")

def test(one,two,three):
    s = "第一个{one},第二个{two},第三个{three}"
    s = s.format_map(vars())
    print(s)

test(one="apple",two="banana",three="pineapple")