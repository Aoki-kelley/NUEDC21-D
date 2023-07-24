# 服务端代码

import cv2
import sys
import time
import socket
import threading
import numpy as np
from image_handler import img_handler
from calculator import PeriodCalculator, ProbabilityFilterClass

SERVER_IP = "localhost"
PORT_FOR_NodeA = 8001
PORT_FOR_NodeB = 8002

DATA_TIME_LENGTH = 14  # 数据包头时间信息的长度

MEASURE_START_FLAG = False
DISPLAY_MODE = 0  # 显示模式 0-全显示 1-显示节点A 2-显示节点B
WINDOWS_NEED_DESTROY = []

NodeA_IMG_BUFFFER = np.ndarray(shape=(400, 600, 3), dtype="uint8")
NodeB_IMG_BUFFFER = np.ndarray(shape=(400, 600, 3), dtype="uint8")

GRAVITY = 9.7985

period_calculator_nodeA: PeriodCalculator = PeriodCalculator(11, 3, 5)  # 节点A 周期计算器
period_calculator_nodeB: PeriodCalculator = PeriodCalculator(11, 3, 5)  # 节点B 周期计算器
# 计算结果记录,最多五条,用于取平均值或中位数
period_result_nodeA: list = [0.0]
period_result_nodeB: list = [0.0]
extremum_length_nodeA: list = [0.0]
extremum_length_nodeB: list = [0.0]


class ServerUDP:
    def __init__(self, bind_ip, bind_port):
        """初始化函数"""
        self.DATA_HEADER_LENGTH = 10
        self.RECEIVE_DATA_MAX_LENGTH = 300000
        try:
            self.machine = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
            self.machine.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.machine.bind((bind_ip, bind_port))
        except Exception as e:
            print("client init fail: ", repr(e))

    def __str__(self):
        return "Socket ClientUDP"

    def wait_data_from_client(self, func=None):
        """等待数据 接收的数据格式为定长的数据长度信息+数据 func为可指定的处理数据的函数"""
        while True:
            data_from_client, client_info = self.machine.recvfrom(self.RECEIVE_DATA_MAX_LENGTH)
            # print(time.time(), end="|")
            # print("receive data from {}".format(client_info))
            if client_info:
                self.data_handler(data_from_client, func)

    def data_handler(self, data, func=None):
        try:
            data_size = int(data[0:self.DATA_HEADER_LENGTH])
            receive_buf = b""
            if data_size:
                receive_buf = data[self.DATA_HEADER_LENGTH:]
            if func is not None:
                func(receive_buf)
            else:
                print(receive_buf)

        except Exception as e:
            print("receive fail: ", repr(e))


def data_handler_nodeA(_data):
    global NodeA_IMG_BUFFFER, period_calculator_nodeA, period_result_nodeA, extremum_length_nodeA

    sample_time = int(_data[:DATA_TIME_LENGTH])  # 毫秒
    # print("nodeA: ", sample_time)
    img = cv2.imdecode(np.fromstring(_data[DATA_TIME_LENGTH:], np.uint8), cv2.IMREAD_COLOR)
    NodeA_IMG_BUFFFER, target_pos = img_handler(img)  # 描绘边框及获取目标位置
    if target_pos[0] is not None:  # 计算和记录周期
        cal_period = period_calculator_nodeA.calculate([target_pos[0], sample_time])
        if len(period_result_nodeA) < 5:
            period_result_nodeA.append(cal_period)
        else:
            period_result_nodeA = period_result_nodeA[1:]
            period_result_nodeA.append(cal_period)

        extremum_length = period_calculator_nodeA.get_extremum_length()
        if len(extremum_length_nodeA) < 5:  # 记录最远到最近的距离,用于计算角度
            extremum_length_nodeA.append(extremum_length)
        else:
            extremum_length_nodeA = extremum_length_nodeA[1:]
            extremum_length_nodeA.append(extremum_length)


def data_handler_nodeB(_data):
    global NodeB_IMG_BUFFFER, period_calculator_nodeB, period_result_nodeB, extremum_length_nodeB

    sample_time = int(_data[:DATA_TIME_LENGTH])  # 毫秒
    # print("nodeB: ", sample_time)
    img = cv2.imdecode(np.fromstring(_data[DATA_TIME_LENGTH:], np.uint8), cv2.IMREAD_COLOR)
    NodeB_IMG_BUFFFER, target_pos = img_handler(img)  # 描绘边框及获取目标位置
    if target_pos[0] is not None:  # 计算和记录周期
        cal_period = period_calculator_nodeB.calculate([target_pos[0], sample_time])
        if len(period_result_nodeB) < 5:
            period_result_nodeB.append(cal_period)
        else:
            period_result_nodeB = period_result_nodeB[1:]
            period_result_nodeB.append(cal_period)

        extremum_length = period_calculator_nodeB.get_extremum_length()
        if len(extremum_length_nodeB) < 5:  # 记录最远到最近的距离,用于计算角度
            extremum_length_nodeB.append(extremum_length)
        else:
            extremum_length_nodeB = extremum_length_nodeB[1:]
            extremum_length_nodeB.append(extremum_length)


def receive_video_from_client_nodeA():
    print("wait nodeA...")
    server_udp = ServerUDP(SERVER_IP, PORT_FOR_NodeA)
    server_udp.wait_data_from_client(data_handler_nodeA)


def receive_video_from_client_nodeB():
    print("wait nodeB...")
    server_udp = ServerUDP(SERVER_IP, PORT_FOR_NodeB)
    server_udp.wait_data_from_client(data_handler_nodeB)


def length_calculate(_period):
    """参数为周期 单位为秒"""
    ret = (_period ** 2 * GRAVITY) / (4 * np.pi ** 2) ** 0.5
    return ret


def main():
    global DISPLAY_MODE, NodeA_IMG_BUFFFER, NodeB_IMG_BUFFFER, WINDOWS_NEED_DESTROY
    global period_result_nodeA, period_result_nodeB, extremum_length_nodeA, extremum_length_nodeB

    thread_node_a = threading.Thread(target=receive_video_from_client_nodeA)
    thread_node_b = threading.Thread(target=receive_video_from_client_nodeB)
    thread_node_a.start()
    thread_node_b.start()

    probability_filter = ProbabilityFilterClass(1.5, 0.5, 0.02)

    while True:
        time_start = time.time()

        key = cv2.waitKey(1)
        if key == ord("o"):  # 全部显示
            DISPLAY_MODE = 0
        elif key == ord("a"):  # 显示节点A
            DISPLAY_MODE = 1
        elif key == ord("b"):  # 显示节点B
            DISPLAY_MODE = 2
        elif key == ord("q"):  # 关闭
            cv2.destroyAllWindows()
            print("end")
            sys.exit()

        if DISPLAY_MODE == 0:
            WINDOWS_NEED_DESTROY = ["Video NodeA", "Video NodeB"]
            cv2.imshow("Video Both", cv2.hconcat([NodeA_IMG_BUFFFER, NodeB_IMG_BUFFFER]))
        elif DISPLAY_MODE == 1:
            WINDOWS_NEED_DESTROY = ["Video Both", "Video NodeB"]
            cv2.imshow("Video NodeA", NodeA_IMG_BUFFFER)
        elif DISPLAY_MODE == 2:
            WINDOWS_NEED_DESTROY = ["Video Both", "Video NodeA"]
            cv2.imshow("Video NodeB", NodeB_IMG_BUFFFER)

        for _w in WINDOWS_NEED_DESTROY:
            if cv2.getWindowProperty(_w, cv2.WND_PROP_VISIBLE) > 0:  # 检测窗口是否开启
                cv2.destroyWindow(_w)

        cal_period = (np.mean(period_result_nodeA) + np.mean(period_result_nodeB)) / 2  # 平均值
        # cal_period = (np.median(period_result_nodeA) + np.median(period_result_nodeB)) / 2  # 中位数
        cal_length = length_calculate(cal_period) - 0.141  # 误差矫正
        extremum_length_a = np.mean(extremum_length_nodeA)  # 平均值
        extremum_length_b = np.mean(extremum_length_nodeB)
        # extremum_length_a = np.median(extremum_length_nodeA)  # 中位数
        # extremum_length_b = np.median(extremum_length_nodeB)

        if extremum_length_a == 0:  # 角度计算
            angle = 0
        elif extremum_length_b == 0:
            angle = 90
        else:
            angle = np.arctan(extremum_length_a / extremum_length_b) / np.pi * 180

        cal_length = probability_filter.calculate(cal_length)  # 概率滤波
        cal_length = round(cal_length, 3)  # 精度为毫米

        # print("period: ", cal_period)
        print("cal_length: ", cal_length)
        # print("extremum_length_a: ", extremum_length_a)
        # print("extremum_length_b: ", extremum_length_b)
        print("angle: ", angle)
        print("time: ", time.time() - time_start)
        print("--------")


if __name__ == "__main__":
    while True:
        tmp = input(">")
        if tmp == "1":  # 启动
            print("start!")
            MEASURE_START_FLAG = True
            break

    if MEASURE_START_FLAG:
        main()
