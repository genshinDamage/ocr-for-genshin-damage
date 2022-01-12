import cv2
import numpy as np
import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import xlwt
import xlrd
from tkinter import filedialog, Tk
import pandas as pd


def video5diff(path, video, damage_type):
    vc = cv2.VideoCapture(video)  # 读入视频文件
    c = 0
    rval = vc.isOpened()
    frame_num = 0
    previous = np.ones((1408, 864))
    kernel = np.ones((3, 3), np.uint8)
    while rval:  # 循环读取视频帧
        c = c + 1
        rval, frame = vc.read()
        if rval:
            frame = frame[86:950, 242:1650]  # 裁剪无关区域
            result = frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, element_low[damage_type], element_high[damage_type])
            dilate = cv2.dilate(mask, kernel)
            result[dilate == 0] = (0, 0, 0)
            result2 = result.copy()
            if frame_num:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, previous)
                dilate = cv2.dilate(diff, kernel)
                _, mask = cv2.threshold(dilate, 30, 255, cv2.THRESH_BINARY)
                result2[mask == 0] = (0, 0, 0)
            cv2.imwrite(path + '/pre_process/' + str(damage_type+1) + "_" + str(c) + '.jpg',
                        result2)  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
            previous = result
            frame_num += 1
        else:
            break
    vc.release()


def batch_processing(directory, video, damage_type):  # ./sth/
    os.makedirs(directory + '/pre_process')
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('sheet1')
    column = ["order", "length", "height", "damage", "damage_type", "x", "y", "credibility"]
    for i, j in enumerate(column):
        worksheet.write(0, i, j)
    k = 1
    for i in damage_type:
        video5diff(directory, video, i)
    image = os.listdir(directory + '/pre_process/')  # sth2.jpg
    for image_name in image:
        result = img_ocr(directory + '/pre_process/' + image_name)
        location = [line[0] for line in result]
        number = [line[1][0] for line in result]
        credibility = [line[1][1] for line in result]
        damage_type = image_name[0]
        upleft_point = [line[0][0] for line in result]
        for i, j in enumerate(number):
            if isvalid(location[i], number[i], credibility[i]):
                worksheet.write(k, 0, image_name.split("_")[-1].split(".")[0])  # 第一列几帧
                worksheet.write(k, 1, location[i][1][0] - location[i][0][0])  # 第二列框长
                worksheet.write(k, 2, location[i][2][1] - location[i][0][1])  # 第三列框高
                worksheet.write(k, 3, number[i])  # 第四列伤害(str，存在以0开头的数字,需进一步处理）
                worksheet.write(k, 4, damage_type)  # 第五列伤害类型
                worksheet.write(k, 5, upleft_point[i][0])  # 第六，七列顶点横纵坐标
                worksheet.write(k, 6, upleft_point[i][1])
                k = k + 1
    workbook.save(directory + '/pre_process/' + "pre_process.xls")


def img_ocr(file):
    ocr = PaddleOCR(use_angle_cls=False, lang="ch")
    result = ocr.ocr(file, cls=False)  # 全是正向数字，不需要
    return result


element_low = [(132, 82, 213), (161, 76, 185), (10, 197, 155), (0, 0, 229), (94, 140, 219), (67, 105, 218),
               (9, 59, 186), (87, 61, 237)]
element_high = [(150, 156, 255), (179, 140, 255), (20, 255, 255), (163, 12, 255), (105, 213, 255), (85, 255, 255),
                (24, 164, 255), (106, 117, 255)]


def isrect(vertex):
    if abs(vertex[0][0] - vertex[3][0]) < 6 and abs(vertex[0][1] - vertex[1][1]) < 6 and \
            abs(vertex[1][0] - vertex[2][0]) < 6 and abs(vertex[2][1] - vertex[3][1]) < 6:  # 正常识别框横平竖直
        return True


def isvalid(location, number, credibility):
    if credibility > 0.9 and number.isdigit() and isrect(location):
        return True


def main():
    root = Tk()
    root.withdraw()  # 隐藏主窗口，只弹出打开文件对话框
    video_file = filedialog.askopenfilename()  # ./sth.mp4
    input_str = input("请输入队伍可以造成以下哪些伤害，用逗号隔开\n1. 雷 2. 超载\n3. 火 4. 物理\n5. 水 6. 风\n7. 岩 8. 冰\n")
    damage_list = [int(i) - 1 for i in input_str.split(',')]
    video_path = os.path.split(video_file)[0]  # .
    batch_processing(video_path, video_file, damage_list)


if __name__ == '__main__':
    main()
