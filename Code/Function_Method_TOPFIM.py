import urllib.parse
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
from numba import jit
from netCDF4 import Dataset
import uvicorn
import traceback
from fastapi import FastAPI, status, UploadFile, File
from fastapi.responses import JSONResponse, Response
from typing import Union
import json
from osgeo import gdal
import os
# from Manning_Calculate import discharge_calculation
from matplotlib.animation import FuncAnimation
import pandas as pd
import math
from operator import itemgetter

app = FastAPI()
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 自定义返回函数
def reponse(*, code, message, TimeElasped='0', data: Union[list, dict, str] = '') -> Response:
    return JSONResponse(
        # status_code=status.HTTP_200_OK,
        content={
            'code': code,
            'message': message,
            'data': data,
        },
        headers={
            'mversion': "1.0",
            'code': code,
            'message': message,
            'TimeElasped': TimeElasped
        }
    )


class BaseSection(object):
    def breadth(self, h: float):
        raise NotImplementedError('breadth 方法必须被重写')

    def area(self, h: float):
        raise NotImplementedError('area 方法必须被重写')

    def perimeter(self, h: float):
        raise NotImplementedError('perimeter 方法必须被重写')

    def radius(self, h: float):
        return self.area(h) / self.perimeter(h)

    def element(self, h: float):
        return {
            'h': h,
            'B': self.breadth(h),  # 水面宽
            'A': self.area(h),  # 过水断面面积
            'X': self.perimeter(h),  # 湿周
            'R': self.radius(h),  # 水力半径
        }

    def manning(self, h: float, n: float, j: float):
        if not hasattr(self, 'element'):
            raise NotImplementedError('element 方法必须被定义')
        element = self.element(h)
        R = element.get("R")
        A = element.get("A")
        C = 1 / n * R ** (1 / 6)
        V = C * math.sqrt(R * j)
        Q = A * V
        return {
            **element,
            "C": C,
            "V": V,
            "Q": Q,
        }


class TrapezoidalSection(BaseSection):
    def __init__(self, m: float, b: float):
        self.m = m
        self.b = b

    def area(self, h: float):
        return h * (self.b + self.m * h)

    def breadth(self, h: float):
        return self.b + 2 * h * self.m

    def perimeter(self, h: float):
        self.b + 2 * h * math.pow(1 + self.m ** 2, 0.5)


class DuplexSection(BaseSection):
    def __init__(self, m1: float, m2: float, b1: float, b2: float, h1: float):
        self.m1 = m1
        self.m2 = m2
        self.b1 = b1
        self.b2 = b2
        self.h1 = h1

    def breadth(self, h: float):
        if h <= self.h1:
            return self.b1 + 2 * self.m1 * h
        return self.b2 + 2 * self.m2 * (h - self.h1)

    def area(self, h: float):
        if h <= self.h1:
            return h * (self.b1 + self.m1 * h)
        return self.h1 * (self.b1 + self.m1 * self.h1) + (h - self.h1) * \
            (self.b2 + self.m2 * (h - self.h1))

    def perimeter(self, h: float):
        if h <= self.h1:
            return self.b1 + 2 * h * math.sqrt(1 + self.m1 ** 2)
        return self.b2 - 2 * self.m1 * self.h1 + 2 * self.h1 * math.sqrt(1 + self.m1 ** 2) \
            + 2 * (h - self.h1) * math.sqrt(1 + self.m2 ** 2)


class USection(BaseSection):
    def __init__(self, r: float, m: float):

        self.r = r
        self.m = m
        self.theta = 2 * math.atan2(1, m)
        self.b = 2 * r / math.sqrt(1 + m ** 2)
        self.h1 = r * (1 - m / math.sqrt(1 + m ** 2))

    def alpha(self, h: float):
        return 2 * math.acos((self.r - h) / self.r)

    def area(self, h: float):
        if h <= self.h1:
            return self.r ** 2 * (self.alpha(h) - math.sin(self.alpha(h))) / 2
        return self.r ** 2 * (self.theta - self.m / (1 + self.m ** 2)) / 2 \
            + (self.b + 2 * self.m * (h - self.h1)) * (h - self.h1)

    def perimeter(self, h: float):
        if h <= self.h1:
            return self.r * self.alpha(h)
        return self.r * self.theta + 2 * (h - self.h1) * math.sqrt(1 + self.m ** 2)

    def breadth(self, h: float):
        if h <= self.h1:
            return 2 * math.sqrt(2 * h * self.r - h ** 2)
        return self.b + 2 * self.m * (h - self.h1)


class CircleSection(BaseSection):

    def __init__(self, r: float):
        self.r = r

    def theta(self, h: float):
        return 2 * math.acos((self.r - h) / self.r)

    def breadth(self, h: float):
        return 2 * math.sqrt(h * (2 * self.r - h))

    def area(self, h: float):
        theta = self.theta(h)
        self.r ** 2 * (theta - math.sin(theta)) / 2

    def perimeter(self, h: float):
        return self.r * self.theta(h)


class ParabolaSection(BaseSection):
    def __init__(self, h: float, b: float):
        self.h = h
        self.b = b

    def area(self, h: float):
        return 2 / 3 * self.breadth(h) * h

    def perimeter(self, h: float):
        return math.sqrt((1 + 4 * h) * h) + 0.5 * math.log(
            2 * math.sqrt(h) + math.sqrt(1 + 4 * h), math.e)

    def breadth(self, h: float):
        return self.b * math.sqrt(h / self.h)


class MeasuredSection(BaseSection):
    def __init__(self, coords):

        self.coords = sorted(coords, key=itemgetter(0))

    def area(self, h: float):
        return self.element(h).get('A')

    def perimeter(self, h: float):
        return self.element(h).get('X')

    def breadth(self, h: float):
        return self.element(h).get('B')

    def element(self, h: float):
        x, y = list(zip(*self.coords))
        if h < min(y):
            print('水位低于河底！')
            raise ValueError
        if h > max(y):
            print('水位高于堤顶！')
            raise ValueError
        s = 0
        ka = 0
        b = 0
        for i in range(0, len(x) - 1):
            if y[i] != y[i + 1]:
                x0 = (h - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]) + x[i]
            else:
                x0 = x[i + 1]
            s1 = (h - y[i + 1]) * (x[i + 1] - x0) / 2
            s2 = (h - y[i]) * (x0 - x[i]) / 2
            s3 = (2 * h - y[i] - y[i + 1]) * (x[i + 1] - x[i]) / 2
            ka1 = ((x[i + 1] - x0) ** 2 + (y[i + 1] - h) ** 2) ** 0.5
            ka2 = ((x[i] - x0) ** 2 + (y[i] - h) ** 2) ** 0.5
            ka3 = ((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2) ** 0.5
            b1 = x[i + 1] - x0
            b2 = x0 - x[i]
            b3 = x[i + 1] - x[i]
            if y[i] >= h > y[i + 1] or y[i] > h >= y[i + 1]:
                s += s1
                ka += ka1
                b += b1
            elif y[i] <= h < y[i + 1] or y[i] < h <= y[i + 1]:
                s += s2
                ka += + ka2
                b += b2
            elif h > y[i] and h > y[i + 1]:
                s += s3
                ka += ka3
                b += b3

        return {
            'h': h - min(y),
            'B': b,
            'A': s,
            'X': ka,
            'R': s / ka if ka != 0 else 0,
        }


def discharge_calculation(file_path, water_level, manning_n, slope):
    section_data = pd.read_table(file_path)
    section_data = list(zip(section_data['X'], section_data['Graphic Profile 1']))
    section = MeasuredSection(section_data)
    section_dict = section.element(water_level)
    discharge = (1 / manning_n) * section_dict['A'] * (section_dict['R'] ** (2 / 3)) * (slope ** 0.5)
    return discharge


@jit(nopython=True)
def get_RiverPoint(RiverNet, NoD_riv):
    row, col = RiverNet.shape
    river_point = []
    for i in range(row):
        for j in range(col):
            if RiverNet[i, j] != NoD_riv:
                river_point.append((i, j))
    return river_point


def HAN_get(FlowDir, RiverNet, NoData):
    row, col = FlowDir.shape

    river_point = get_RiverPoint(RiverNet, NoData)
    print('河流栅格点已准备就绪')

    HAN = np.zeros((row, col), dtype='int32')
    HAND = np.zeros((row, col), dtype='float32')

    @jit(nopython=True)
    def reload_HAN(row, col, FlowDir, NoData, HAN, HAND):
        for i in range(row):
            for j in range(col):
                if FlowDir[i, j] == NoData:
                    HAN[i, j] = NoData
                    HAND[i, j] = NoData
        return HAN, HAND

    HAN, HAND = reload_HAN(row, col, FlowDir, NoData, HAN, HAND)

    for i in enumerate(river_point):
        idx, value = i
        HAN[value] = idx + 1

    for i in river_point:
        FlowDir[i] = 0

    @jit(nopython=True)
    def find_river(direction, idx_i, idx_j):
        par_k = 0
        if direction[idx_i, idx_j] == 1:
            if direction[idx_i, idx_j + par_k] != 0:
                par_k += 1
            idx_j = idx_j + par_k
        elif direction[idx_i, idx_j] == 16:
            if direction[idx_i, idx_j - par_k] != 0:
                par_k += 1
            idx_j = idx_j - par_k
        elif direction[idx_i, idx_j] == 4:
            if direction[idx_i + par_k, idx_j] != 0:
                par_k += 1
            idx_i = idx_i + par_k
        elif direction[idx_i, idx_j] == 2:
            if direction[idx_i + par_k, idx_j + par_k] != 0:
                par_k += 1
            idx_i = idx_i + par_k
            idx_j = idx_j + par_k
        elif direction[idx_i, idx_j] == 8:
            if direction[idx_i + par_k, idx_j - par_k] != 0:
                par_k += 1
            idx_i = idx_i + par_k
            idx_j = idx_j - par_k
        elif direction[idx_i, idx_j] == 64:
            if direction[idx_i - par_k, idx_j] != 0:
                par_k += 1
            idx_i = idx_i - par_k
        elif direction[idx_i, idx_j] == 128:
            if direction[idx_i - par_k, idx_j + par_k] != 0:
                par_k += 1
            idx_i = idx_i - par_k
            idx_j = idx_j + par_k
        elif direction[idx_i, idx_j] == 32:
            if direction[idx_i - par_k, idx_j - par_k] != 0:
                par_k += 1
            idx_i = idx_i - par_k
            idx_j = idx_j - par_k
        return idx_i, idx_j

    @jit(nopython=True)
    def get_HAN(FlowDir, HAN, NoData):
        row, col = FlowDir.shape
        print(row, col)
        errNum = 0
        totalNum = 1
        for i in range(row):
            for j in range(col):
                if FlowDir[i, j] != 0 and FlowDir[i, j] != NoData:
                    k, l = find_river(FlowDir, i, j)

                    if k < 0 or k >= row or l < 0 or l >= col or FlowDir[k, l] == NoData:
                        HAN[i, j] = -2
                        k, l = 0, 0
                    while HAN[i, j] != -2 and FlowDir[k, l] != 0:

                        k, l = find_river(FlowDir, k, l)
                        # print(k, l)
                        if FlowDir[k, l] == NoData or k < 0 or k >= row or l < 0 or l >= col:
                            errNum += 1
                            k, l = 0, 0
                            break
                    if k == 0 and l == 0:
                        HAN[i, j] = -2
                    else:
                        HAN[i, j] = HAN[k, l]
                    totalNum += 1
            if i % 100 == 0 and i != 0:
                print('comp：', round(100 * i / row, 2), '%', 'er：', round(100 * errNum / totalNum, 2),
                      '%')
        return HAN

    HAN = get_HAN(FlowDir, HAN, NoData)

    # HAN = HAN_adjust(HAN, RiverNet, NoData)
    return HAN, HAND, river_point


def HAN_adjust(HAN, RiverNet, NoData):
    row, col = HAN.shape

    @jit(nopython=True)
    def get_RiverPoint(row, col, RiverNet, NoData):
        river_point = []
        for i in range(row):
            for j in range(col):
                if RiverNet[i, j] != NoData:
                    river_point.append((i, j))
        return river_point

    river_point = get_RiverPoint(row, col, RiverNet, NoData)

    @jit(nopython=True)
    def get_errorIdx(HAN):
        row, col = HAN.shape
        HAN_errorList = []
        for i in range(row):
            for j in range(col):
                if HAN[i, j] == -2:
                    HAN_errorList.append((i, j))
        return HAN_errorList

    HAN_errorList = get_errorIdx(HAN)

    @jit(nopython=True)
    def get_min(river_point, idx):
        length_list = []
        for i in river_point:
            length_list.append(abs(idx[0] - i[0]) + abs(idx[1] - i[1]))
        return length_list

    @jit(nopython=True)
    def get_Min(HAN, HAN_errorList, river_point):
        for value, idx in enumerate(HAN_errorList):
            length_list = get_min(river_point, idx)
            idx_min = river_point[length_list.index(min(length_list))]
            HAN[idx] = HAN[idx_min]
            if value % 1000 == 0:
                print('rev:', value, '/', len(HAN_errorList), 'idx:-->', idx_min)
        return HAN

    if len(HAN_errorList) != 0:
        HAN = get_Min(HAN, HAN_errorList, river_point)

    return HAN


# HAND值计算
def HAND_get(river_point, DEM, HAND, HAN, NoData):
    @jit(nopython=True)
    def get_DEMList(river_point, DEM):
        DEM_list = []
        for river in river_point:
            DEM_list.append(DEM[river])
        return DEM_list

    DEM_list = get_DEMList(river_point, DEM)

    @jit(nopython=True)
    def get_HAND(HAN, HAND, DEM, DEM_list, NoData):
        row, col = DEM.shape
        for i in range(row):
            for j in range(col):
                if HAND[i, j] != NoData and HAN[i, j] != -2:
                    HAND[i, j] = DEM[i, j] - DEM_list[HAN[i, j] - 1]
                    if HAND[i, j] < NoData == -9999:
                        HAND[i, j] = NoData
                    # elif HAND[i, j] < 0 and HAND[i, j] != NoData:
                    #     HAND[i, j] = DEM[i, j]
        return HAND

    HAND = get_HAND(HAN, HAND, DEM, DEM_list, NoData)

    return HAND


@jit(nopython=True)
def set_uplist(riv_HAND, idx_list, HAND, HAN_num):
    row, col = HAND.shape
    HAND_num = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if HAN_num[i, j] != HAN_num[0, 0]:
                HAND_num[i, j] = riv_HAND[int(HAN_num[i, j] - 1)]
    return HAND_num


@jit(nopython=True)
def flood_appear(HAND_num, HAND, NoData):
    row, col = HAND.shape
    flood_level = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if HAND[i, j] != NoData and HAND[i, j] < HAND_num[i, j]:
                flood_level[i, j] = HAND_num[i, j] - HAND[i, j]
    return flood_level


def New_inp(x, y, x2):
    if len(x) >= 5:
        for idx, value in enumerate(x):
            if value >= x2:
                idx_a = idx - 2
                idx_b = idx + 3
                if idx_a < 0:
                    idx_a = 0
                    idx_b = 5
                if idx_b > len(x):
                    idx_a = len(x) - 5
                    idx_b = len(x)
                x = x[idx_a:idx_b]
                y = y[idx_a:idx_b]
                break
    num = len(x) - 1

    mean_diff = y[:]
    mean_list = []

    for j in range(num):
        for i in range(len(x) - j - 1):
            mean_diff.append((mean_diff[i + 1] - mean_diff[i]) / (x[i + j + 1] - x[i]))
        mean_diff = mean_diff[len(x) - j:]
        mean_list.append(mean_diff[0])

    pam = 1
    pami = []
    f = 0
    for i in range(num):
        pam *= x2 - x[i]
        pami.append(pam)
        f += mean_list[i] * pami[i]
    f += y[0]
    return f


@jit(nopython=True)
def get_HAN_num(HAN_num, HAN_array, idx_list):
    row, col = HAN_num.shape
    for i in range(row):
        for j in range(col):
            for k, HAN_k in enumerate(HAN_array):
                if HAN_k[i, j] != 0:
                    HAN_num[i, j] = idx_list[k]
    return HAN_num


@jit(nopython=True)
def get_han_num2(HAN_sig, HAN, HAN_num, han_idx):
    row, col = HAN.shape
    for i in range(row):
        for j in range(col):
            if HAN_sig[i, j] != 0:
                HAN_num[i, j] = han_idx
    return HAN_num


@jit(nopython=True)
def clip_RiverNet(RiverNet, clip_num):
    aim_list = []
    RiverNet_new = np.ones(RiverNet.shape) * -9999
    row, col = RiverNet.shape
    riv_number_idx = 1
    for i in range(row):
        for j in range(col):
            if RiverNet[i, j] == clip_num:
                aim_list.append((i, j))
                RiverNet_new[i, j] = riv_number_idx
                riv_number_idx += 1
    return RiverNet_new


# @jit(nopython=True)
def Control_Area_Division(HAN, RiverNet, NoData):
    print('Control Area Division...')
    row, col = HAN.shape

    river_idx = list(np.unique(RiverNet))
    idx_list = river_idx[1:]

    riv_list_l = [[] for _ in range(len(idx_list))]
    HAN_list_l = [[] for _ in range(len(idx_list))]

    for i in range(row):
        for j in range(col):
            if RiverNet[i, j] != NoData:
                idx_num = idx_list.index(RiverNet[i, j])
                riv_list_l[idx_num].append((i, j))
                HAN_list_l[idx_num].append(HAN[i, j])

    HAN_num = np.ones((row, col)) * -9999
    for idx in range(len(idx_list)):
        print(idx)
        han_arr = np.zeros(HAN.shape)
        HAN_list = HAN_list_l[idx]
        HAN_sig = get_han(han_arr, HAN, HAN_list)
        HAN_num = get_han_num2(HAN_sig, HAN, HAN_num, idx + 1)

    return HAN_num


# 淹没判断
@jit(nopython=True)
def flood_appear(HAND_num, HAND, NoData):
    row, col = HAND.shape
    flood_level = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if HAND[i, j] != NoData and HAND[i, j] < HAND_num[i, j]:
                flood_level[i, j] = HAND_num[i, j] - HAND[i, j]
    return flood_level


@jit(nopython=True)
def DTFV_Curve(HAND, HAN_num, NoData, DT_num):
    print('DTFV Curve...')

    river_idx = list(np.unique(HAN_num))
    idx_list = list(map(int, river_idx[1:]))
    DTFV_row, DTFV_col = len(idx_list), DT_num
    DTFV_list = np.zeros((DTFV_row, DTFV_col))
    flood_number_list = np.zeros((DTFV_row, DTFV_col))
    row, col = HAND.shape

    for DT in range(DT_num):
        print(DT)
        riv_HAND = [float(DT)] * len(idx_list)
        HAND_num = set_uplist(riv_HAND, idx_list, HAND, HAN_num)
        flood_level = flood_appear(HAND_num, HAND, NoData)
        for i in range(row):
            for j in range(col):
                if flood_level[i, j] > 0 and HAN_num[i, j] != NoData:
                    DTFV_list[int(HAN_num[i, j] - 1), int(DT)] += flood_level[i, j]
                    flood_number_list[int(HAN_num[i, j] - 1), int(DT)] += 1
    return DTFV_list, flood_number_list


def s_cal(DEM, river, cellsize, NoData):
    river_idx = list(np.unique(river))  # 去除重复值，按大小排序，第一位为NoData：0需删掉
    idx_list = list(map(int, river_idx[1:]))  # 河段编号
    S_list = []
    river_length_list = []

    for idx in idx_list:
        river_idx = filterate_river(river, idx)  # 提取仅有该段的河网
        river_point_sd = get_RiverPoint(river_idx, NoData)  # 河段栅格编号
        river_length = len(river_point_sd) * cellsize
        river_length_list.append(river_length)
        # S_list.append(abs(DEM[river_point_sd[0]] - DEM[river_point_sd[-1]])/river_length)
        S_each_point = []
        # for point_idx in range(len(river_point_sd) - 1):
        #     S_each_point.append(abs(DEM[river_point_sd[point_idx]] - DEM[river_point_sd[point_idx + 1]]) / cellsize)
        #
        # # S_list.append(round((sum(S_each_point)/len(S_each_point)), 4))
        # S_list.append(sum(S_each_point) / len(S_each_point))
        S_list.append(abs((DEM[river_point_sd[0]] - DEM[river_point_sd[-1]])) / (len(river_point_sd) * cellsize))
    S_list = [s_l if s_l != 0 else min(filter(lambda x: x != 0, S_list)) for s_l in S_list]

    return S_list, river_length_list


def SD_Curve(HAND, HAN_num, NoData, DT_num, S_list, cellsize, slp, n, seg_length_list):
    print('SD Curve...')
    river_idx = list(np.unique(HAN_num))  # 去除重复值，按大小排序，第一位为NoData：0需删掉
    idx_list = list(map(int, river_idx[1:]))  # 河段编号
    SD_row, SD_col = len(idx_list), DT_num
    SD_list = np.zeros((SD_row, SD_col))
    By_list = np.zeros((SD_row, SD_col))
    Vy_list = np.zeros((SD_row, SD_col))
    Ay_list = np.zeros((SD_row, SD_col))
    Q_list = np.zeros((SD_row, SD_col))

    row, col = HAND.shape

    for DT in range(DT_num):
        print(DT)
        riv_HAND = [float(DT)] * len(idx_list)
        HAND_num = set_uplist(riv_HAND, idx_list, HAND, HAN_num)
        flood_level = flood_appear(HAND_num, HAND, NoData)

        for i in range(row):
            for j in range(col):
                if flood_level[i, j] != 0:  # 淹没区
                    By_list[int(HAN_num[i, j] - 1), DT] += cellsize * cellsize * ((1 + (slp[i, j] ** 2)) ** 0.5)
                    Vy_list[int(HAN_num[i, j] - 1), DT] += (flood_level[i, j] * cellsize * cellsize)
    R_y = Vy_list / By_list
    for SD in range(SD_row):
        Q_list[SD, :] = (R_y[SD, :] ** (2 / 3)) * (S_list[SD] ** 0.5) * (Vy_list[SD, :] / seg_length_list[SD]) / n
        Ay_list[SD, :] = Vy_list[SD, :] / seg_length_list[SD]
    return Q_list, Ay_list


@jit(nopython=True)
def find_near_coor(r, c, river, NoData):
    idx_list = [(r - 1, c - 1), (r - 1, c), (r - 1, c + 1), (r, c - 1), (r, c + 1), (r + 1, c - 1), (r + 1, c),
                (r + 1, c + 1)]
    point_idx = []
    row, col = river.shape
    for i, j in idx_list:
        if 0 <= i < row and 0 <= j < col:
            if river[i, j] != NoData:
                point_idx.append((i, j))
    return point_idx


def get_min(river_point, idx):
    length_list = []
    for i in river_point:
        length_list.append(abs(idx[0] - i[0]) + abs(idx[1] - i[1]))
    min_length = min(length_list)
    min_idx = length_list.index(min_length)
    min_value = river_point[min_idx]
    return min_value


def get_max(river_point, idx):
    length_list = []
    for i in river_point:
        length_list.append(abs(idx[0] - i[0]) + abs(idx[1] - i[1]))
    max_length = max(length_list)
    max_idx = length_list.index(max_length)
    max_value = river_point[max_idx]
    return max_value


# @jit(nopython=True)
def search_near_grid(river_point_list, river, NoData):
    river_mid = np.copy(river)

    river_point_list_old = river_point_list[:]
    river_point_list_old = sorted(river_point_list_old, key=lambda x: x[1])
    loop_num = 0
    aim_coor = river_point_list[0]
    # aim_coor = (4440, 5661)
    # print(aim_coor)
    # plt.imshow(river)
    # plt.show()
    # exit()
    river[aim_coor] = 1
    river_mid[aim_coor] = NoData
    river_point_list_old.remove(aim_coor)
    aim_coor_list = []
    aim_coor_list.append(aim_coor)
    number_idx = 2
    aim_plt_array = np.copy(river)

    # plt.ion()

    while len(river_point_list_old) != 0:

        near_point_list = find_near_coor(aim_coor[0], aim_coor[1], river_mid, NoData)
        near_point_list = sorted(near_point_list, key=lambda x: x[0])
        # print(aim_coor)
        # print(near_point_list)

        # aim_plt_array[aim_coor] = 2
        # plt.imshow(aim_plt_array)
        # plt.pause(0.000001)
        # plt.clf()

        # print(near_point_list)
        for aim in aim_coor_list:
            if aim in near_point_list:
                near_point_list.remove(aim)
                aim_coor_list.remove(aim)
        # exit()
        if len(near_point_list) == 1:
            river[near_point_list[0]] = number_idx
            river_mid[near_point_list[0]] = NoData

            number_idx += 1
            river_point_list_old.remove(near_point_list[0])
            aim_coor = near_point_list[0]
            aim_coor_list.append(aim_coor)
            if len(river_point_list_old) == 0:
                break
        elif len(near_point_list) > 1:  # 最近栅格数量大于1，下一次搜索起点为据上次起点最远的栅格
            aim_coor = get_max(near_point_list, aim_coor)
            aim_coor_list.append(aim_coor)
            near_point_list.remove(aim_coor)
            river_point_list_old.remove(aim_coor)

            for near_point in near_point_list:
                if near_point in river_point_list_old:
                    river[near_point] = number_idx
                    river_mid[near_point] = NoData
                    number_idx += 1
                    river_point_list_old.remove(near_point)
                    # aim_coor = near_point_list[-1]
                    if len(river_point_list_old) == 0:
                        break
                else:
                    continue
            river[aim_coor] = number_idx
            river_mid[aim_coor] = NoData
            number_idx += 1
        elif len(near_point_list) == 0:
            if len(river_point_list_old) != 0:
                aim_coor = get_min(river_point_list_old, aim_coor)
                aim_coor_list.append(aim_coor)
                river_point_list_old.remove(aim_coor)
                # near_point_list.remove(aim_coor)

                river[aim_coor] = number_idx
                river_mid[aim_coor] = NoData
                number_idx += 1

            else:
                print('error')
                break
        loop_num += 1
    # plt.ioff()
    # plt.show()
    # print(len(river_point_list_old))
    return river


@jit(nopython=True)
def filterate_river(RiverNet, aim_number):
    row, col = RiverNet.shape
    RiverNet_new = np.ones(RiverNet.shape) * -9999
    for i in range(row):
        for j in range(col):
            if RiverNet[i, j] == aim_number:
                RiverNet_new[i, j] = aim_number
    return RiverNet_new


@jit(nopython=True)
def set_nodata(array, nodata):
    row, col = array.shape
    for i in range(row):
        for j in range(col):
            if array[i, j] == nodata:
                array[i, j] = -9999
    return array


def river_segment_sd(river, cellsize, seg_length):
    seg_len = int(seg_length / cellsize)
    seg_num = int(np.max(river) / seg_len)
    # print(seg_num)
    seg_num_list = [seg_len * seg_idx for seg_idx in range(seg_num)]

    print(seg_num_list)

    for idx in range(len(seg_num_list) - 1):
        river = np.where((river >= seg_num_list[idx]) & (river < seg_num_list[idx + 1]), idx + 1, river)
    river = np.where(river >= seg_num_list[-1], idx + 2, river)
    return river


def river_segment_dtfv(river, seg_num_list):
    for idx in range(len(seg_num_list) - 1):
        river = np.where((river >= seg_num_list[idx]) & (river < seg_num_list[idx + 1]), idx + 1, river)
    river = np.where(river >= seg_num_list[-1], idx + 2, river)
    return river


@jit(nopython=True)
def catchment_volume(HAN_num, flood_extent_matrix, idx_list, NoData):
    row, col = HAN_num.shape
    volume_list = [0.] * len(idx_list)
    for i in range(row):
        for j in range(col):
            if flood_extent_matrix[i, j] > 0 and HAN_num[i, j] != NoData:
                volume_list[int(HAN_num[i, j] - 1)] += flood_extent_matrix[i, j]
    return volume_list


def linear_interpolation(x, x_values, y_values):
    n = len(x_values)
    if n != len(y_values):
        raise ValueError("x_values 和 y_values 的长度必须相等")

    i = 0
    while i < n and x_values[i] < x:
        i += 1

    if i == 0:
        return y_values[0]
    elif i == n:
        return y_values[n - 1]

    x0, x1 = x_values[i - 1], x_values[i]
    y0, y1 = y_values[i - 1], y_values[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def read_tif(data_path):
    dataset_tif = gdal.Open(data_path)
    array_tif = dataset_tif.GetRasterBand(1).ReadAsArray()
    array_tif = array_tif.astype(np.float32)
    return array_tif


def array2tif(tif_path, tif_output_path, output_array, NoData=-9999, output_type=gdal.GDT_Float32):
    dataset_tif = gdal.Open(tif_path)
    GeoTransform = dataset_tif.GetGeoTransform()
    GetProjection = dataset_tif.GetProjection()
    driver = gdal.GetDriverByName('GTiff')
    row, col = output_array.shape
    dst_ds = driver.Create(tif_output_path, col, row, 1, output_type)  # 创建数据集
    dst_ds.SetGeoTransform(GeoTransform)
    dst_ds.SetProjection(GetProjection)
    dst_ds.GetRasterBand(1).WriteArray(output_array)
    dst_ds.GetRasterBand(1).SetNoDataValue(NoData)
