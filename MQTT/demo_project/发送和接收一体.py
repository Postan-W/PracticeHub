import gin
from paho.mqtt import client as mqtt_client
import random
import time
#这里是将发送和接受的角色写到一个类中，用到哪个运行哪个
client_id = 'python-mqtt-{}'.format(random.randint(0, 100))

def connect_mqtt(host,port=1883) -> mqtt_client:
    # 连接MQTT服务器
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    # broker = 'broker.emqx.io'
    # port = 1883
    # client.connect(broker, port)
    client.connect(host=host, port=port)
    return client

def subscribe(client: mqtt_client,topic):
    def on_message(client, userdata, msg):
        data = msg.payload.decode()
        print('订阅【{}】的消息为：{}'.format(msg.topic, data))
    client.subscribe(topic)
    client.on_message = on_message
    client.loop_forever()

def publish(client,topic):
    # 发布消息
    msg_count = 0
    while True:
        time.sleep(1)
        msg = '这是客户端发送的第{}条消息'.format(msg_count)
        result = client.publish(topic, msg)
        status = result[0]
        if status == 0:
            print('第{}条消息发送成功'.format(msg_count))
        else:
            print('第{}条消息发送失败'.format(msg_count))
        msg_count += 1

@gin.configurable
class MQTTClient:
    def __init__(self,host,port,topic):
        self.topic = topic
        self.host = host
        self.port = port
        self.client = connect_mqtt(self.host,self.port)

    #订阅
    def self_subscribe(self):
        subscribe(self.client,self.topic)

    #发布
    def self_publish(self):
        publish(self.client,self.topic)

if __name__ == '__main__':
    gin.parse_config_file("config.gin")
    client = MQTTClient()
    client.client.loop_start()
    client.self_subscribe()
    client.self_publish()#同步的。只能运行发送和接受者其中的一个
