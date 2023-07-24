# 计算及滤波

import random


class KalmanFilterOneDimension:
    def __init__(self):
        self.X = 0  # 最优估计值
        self.A = 1  # 状态转移系数
        self.H = 1  # 测量系数
        self.R = 1  # 测量噪声方差
        self.Q = 0.2  # 预测噪声方差
        self.W = 0  # 测量噪声均值
        self.B = 0  # 控制系数
        self.U = 0  # 控制量
        self.P = 0  # 估计值协方差
        self.K = 0  # kalman增益系数

    def generate(self, _z):
        # 迭代 z为测量值
        _x = self.A * self.X + self.B * self.U + self.W  # 估计值
        _p = self.A * self.P * self.A + self.Q  # 估计值协方差
        x_n = _x + self.K * (_z - self.H * _x)  # 最优估计值
        self.X = x_n  # 记录最优估计值
        self.K = _p * self.H / (self.H * _p * self.H + self.R)  # kalman增益
        self.P = (1 - self.K * self.H) * _p  # 最优估计值协方差

        return x_n


class ProbabilityFilterClass:
    def __init__(self, probable_max_val, probable_min_val, scale):
        """
        :param probable_max_val: 预计最大值
        :param probable_min_val: 预计最小值
        :param scale: 最大值到最小值区间的分度值
        """
        self.probable_max_val = probable_max_val
        self.probable_min_val = probable_min_val
        self.scale = scale
        self.range_num = int((probable_max_val - probable_min_val) / scale)  # 区间数量

        self.range_list = [[probable_min_val + i * scale, probable_min_val + (i + 1) * scale] for i in
                           range(self.range_num)]  # 数值区间表,左小右大,左闭右开
        self.range_element_count = [0 for _ in range(self.range_num)]  # 落入区间内的数值数,小标与数值区间表对应
        self.real_val_probability_preference_list = [0.5 for _ in range(self.range_num)]  # 实际值偏向(以左边为准)

    def calculate(self, _val):
        """根据输入数值计算估计值"""
        for i in range(self.range_num):
            tmp_range = self.range_list[i]
            if tmp_range[0] <= _val < tmp_range[1]:  # 在区间范围内
                self.range_element_count[i] += 1  # 区间元素个数加一
                self.real_val_probability_preference_list[i] = 1 - (_val - tmp_range[0]) / self.scale  # 偏向更新

        max_count_index_list = [i for i, x in enumerate(self.range_element_count) if x == max(self.range_element_count)]
        probable_real_val_list = []
        for i in range(len(max_count_index_list)):
            val_range = self.range_list[max_count_index_list[i]]
            probability_preference = self.real_val_probability_preference_list[max_count_index_list[i]]
            probable_real_val_list.append(
                val_range[0] * probability_preference + val_range[1] * (1 - probability_preference))

        if probable_real_val_list:
            return sum(probable_real_val_list) / len(probable_real_val_list)
        else:
            return 0


def sample_simulator() -> list:
    # 取样模拟
    generate_times = 200
    period = 20
    max_val = 300
    min_val = 200
    direction_flag = 1
    ret = [max_val]

    for i in range(generate_times):
        if i % int(period / 2) == 0:
            direction_flag = -direction_flag

        ret.append(ret[-1] + (max_val - min_val) / int(period / 2) * direction_flag + random.randint(-2, 2))
        # ret.append(ret[-1] + (max_val - min_val) / int(period / 2) * direction_flag)

    return ret


class PeriodCalculator:
    """周期计算"""

    def __init__(self, history_record_max, extremum_judge_limit, calculate_accuracy):
        self.history_record_max = history_record_max  # 记录的最大条数,应为奇数,中间点为极值时当作最高点或最低点
        self.extremum_judge_limit = extremum_judge_limit  # 影响极值判断
        self.kalman_filter = KalmanFilterOneDimension()
        self.calculate_accuracy = calculate_accuracy  # 计算精度(小数位数)

        self.pos_time_history = []  # 摆位置及时间记录,用于判断摆的最高点及最低点
        self.highest_pos_time = [0, 0]  # 摆最高位置及到达时间
        self.lowest_pos_time = [0, 0]  # 摆最低位置及到达时间

    def __str__(self):
        return "Period Calculator"

    def get_extremum_length(self):
        return abs(self.highest_pos_time[0] - self.lowest_pos_time[0])

    def calculate(self, _pos_time: list[int, int]) -> float:
        """参数为当前获取的位置及取样时间"""
        if len(self.pos_time_history) < self.history_record_max:
            self.pos_time_history.append(_pos_time)
            return 0
        else:
            self.pos_time_history = self.pos_time_history[1:]
            self.pos_time_history.insert(self.history_record_max - 1, _pos_time)

        middle_index = int((self.history_record_max + 1) / 2) - 1
        middle_val = self.pos_time_history[middle_index][0]
        middle_val_time = self.pos_time_history[middle_index][1]
        left_more_than_middle_val_count = 0
        left_less_than_middle_val_count = 0
        right_more_than_middle_val_count = 0
        right_less_than_middle_val_count = 0
        for j in range(middle_index):
            if self.pos_time_history[j][0] > middle_val:
                left_more_than_middle_val_count += 1
            elif self.pos_time_history[j][0] < middle_val:
                left_less_than_middle_val_count += 1

            if self.pos_time_history[-j][0] > middle_val:
                right_more_than_middle_val_count += 1
            elif self.pos_time_history[-j][0] < middle_val:
                right_less_than_middle_val_count += 1

        if left_more_than_middle_val_count > left_less_than_middle_val_count + self.extremum_judge_limit and \
                right_more_than_middle_val_count > right_less_than_middle_val_count + self.extremum_judge_limit:
            self.lowest_pos_time = [middle_val, middle_val_time]
        elif left_more_than_middle_val_count + self.extremum_judge_limit < left_less_than_middle_val_count and \
                right_more_than_middle_val_count + self.extremum_judge_limit < right_less_than_middle_val_count:
            self.highest_pos_time = [middle_val, middle_val_time]

        cal_period = self.kalman_filter.generate(abs(self.highest_pos_time[1] - self.lowest_pos_time[1]) * 2)
        return round(cal_period / 1000, self.calculate_accuracy)


def main():
    sample_val = sample_simulator()
    period_calculator = PeriodCalculator(9, 2, 5)
    for i in range(len(sample_val)):
        result = period_calculator.calculate([sample_val[i], i])
        print("cal_period: ", result)
        print("res: ", 1 - abs(result - 20) / 20)


if __name__ == "__main__":
    main()
