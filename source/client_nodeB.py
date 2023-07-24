# 节点B 代码

import cv2
import time
import socket
import numpy as np

SERVER_IP = "localhost"
PORT_FOR_NodeB = 8002

DATA_TIME_LENGTH = 14


class ClientUDP:
    def __init__(self):
        """初始化函数"""
        self.DATA_HEADER_LENGTH = 10
        try:
            self.machine = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
            self.machine.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception as e:
            print("client init fail: ", repr(e))

    def __str__(self):
        return "Socket ClientUDP"

    def send_data_to_server(self, server_ip, server_port, data, func=None):
        """发送到服务器 发送的数据格式为定长的数据长度信息+数据 func为可指定的处理数据的函数"""
        print(time.time(), end="|")
        try:
            print("send data to {}".format((server_ip, server_port)))
            data_length_s = ("%{}d".format(self.DATA_HEADER_LENGTH) % len(data)).encode()  # 数据长度长度补偿
            data = data_length_s + data

            if func is not None:
                self.machine.sendto(func(data), (server_ip, server_port))
            else:
                self.machine.sendto(data, (server_ip, server_port))

        except Exception as e:
            print("send fail: ", repr(e))


def main():
    client_udp = ClientUDP()
    capturer = cv2.VideoCapture(1)
    if not capturer.isOpened():
        print("capturer open fail")
        return

    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = capturer.read()
        if ret:
            frame = cv2.resize(frame, (600, 400))
            _, frame_encode = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frame_encode_str = np.array(frame_encode, dtype="uint8").tobytes()
            send_data = ("%{}d".format(DATA_TIME_LENGTH) % (time.time_ns() // 1000000)).encode() + frame_encode_str
            client_udp.send_data_to_server(SERVER_IP, PORT_FOR_NodeB, send_data)
        else:
            capturer.release()
            break

    capturer.release()


if __name__ == "__main__":
    main()
