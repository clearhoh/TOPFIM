import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from sklearn.metrics import cohen_kappa_score


def read_raster(path):
    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    return band.ReadAsArray(), nodata


def read_tif(data_path):
    dataset_tif = gdal.Open(data_path)
    array_tif = dataset_tif.GetRasterBand(1).ReadAsArray()
    array_tif = array_tif.astype(np.float32)
    return array_tif


data_path = './'
HEC_RAS_path = './'

riv = 'jw'
method_list = ['top']
# HAN_num, han_nodata = read_raster(data_path + riv + '.tif')


data1 = read_tif(data_path)
data2 = read_tif(data_path)
if data2.shape == data1.shape:
    row, col = data1.shape
    for i in range(row):
        for j in range(col):
            if data2[i, j] == data2[0, 0]:
                data1[i, j] = data1[0, 0]
    # plt.imshow(data1)
    # plt.show()
    # exit()
else:
    print('sjbyz')
    exit()

comparison_extent = data1
comp_nodata = comparison_extent[0, 0]
hec_ras_extent = read_tif(HEC_RAS_path)
hec_nodata = hec_ras_extent[0, 0]

for method in method_list:
    # hec_ras_extent, hec_nodata = read_raster(data_path + 'hec-' + riv + '.tif')
    # comparison_extent, comp_nodata = read_raster(data_path + method + '-' + riv + '.tif')
    if hec_ras_extent.shape == comparison_extent.shape:
        pass
    else:
        print('sjbyz')
        print(hec_ras_extent.shape, comparison_extent.shape)
        exit()

    row, col = hec_ras_extent.shape
    # print(han_nodata, hec_nodata, comp_nodata)
    # plt.imshow(HAN_num)
    # plt.show()

    # for i in range(row):
    #     for j in range(col):
    #         if HAN_num[i, j] == han_nodata:
    #             if hec_ras_extent[i, j] != hec_nodata:
    #                 hec_ras_extent[i, j] = hec_nodata
    #             if comparison_extent[i, j] != comp_nodata:
    #                 comparison_extent[i, j] = comp_nodata
    # plt.imshow(comparison_extent)
    # plt.show()

    hec_ras_extent = np.where(hec_ras_extent == hec_nodata, 0, 1)
    comparison_extent = np.where(comparison_extent == comp_nodata, 0, 1)

    hec_ras_extent_1d = [item for sublist in hec_ras_extent for item in sublist]
    comparison_extent_1d = [item for sublist in comparison_extent for item in sublist]

    hec_nodata = 0
    comp_nodata = 0

    hec_ras_dry = np.sum(hec_ras_extent == hec_nodata)  # FP+TN
    hec_ras_wet = np.sum(hec_ras_extent != hec_nodata)  # TP+FN
    comparison_wet = np.sum(comparison_extent != comp_nodata)  # TP+FP
    # print(hec_ras_dry)
    # print(hec_ras_wet)
    # print(comparison_wet)

    both_wet = 0  # a
    both_dry = 0  # d
    for i in range(row):
        for j in range(col):
            if hec_ras_extent[i, j] != hec_nodata and comparison_extent[i, j] != comp_nodata:
                both_wet += 1
            elif hec_ras_extent[i, j] == hec_nodata and comparison_extent[i, j] == comp_nodata:
                both_dry += 1

    TP = np.longlong(both_wet)
    TN = np.longlong(both_dry)
    FN = np.longlong(hec_ras_wet - TP)
    FP = np.longlong(hec_ras_dry - TN)

    PC = (TP + TN) / (TP + FN + FP + TN)
    OE = FP / (TP + FP)
    CE = FN / (TP + FN)
    # B = (TP + FP)/(TP+FN)
    # print(type(TP))
    # PE1 = (TP + FP) * (TP + FN)
    # PE2 = (FN + TN) * (FP + TN)
    # PE3 = (TP + FN + TN + FP) ** 2
    # PE4 = PE1+PE2
    # PE = PE4/PE3
    PE = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (TP + FN + TN + FP) ** 2
    K = (PC - PE) / (1. - PE)

    H = TP / (TP + FN)  # Hit Rate

    F = TP / (TP + FN + FP)  # Fitness Statistics
    kappa = cohen_kappa_score(hec_ras_extent_1d, comparison_extent_1d)
    # print(kappa, H, F)
    print(PC, CE, OE)
    # print(kappa)
