import cv2
import numpy as np
import os
from paddleocr import PaddleOCR
import xlwt
from tkinter import filedialog, Tk
import pandas as pd


def video2img(path, name, video):
    os.makedirs(path + '/' + name)
    vc = cv2.VideoCapture(video)  # 读入视频文件
    c = 0
    rval = vc.isOpened()
    while rval:  # 循环读取视频帧
        c = c + 1
        rval, frame = vc.read()
        if rval:
            frame = frame[86:950, 242:1650]  # 裁剪无关区域
            pic_path = path + '/'
            cv2.imwrite(pic_path + name + '/_' + str(c) + '.jpg', frame)
            cv2.waitKey(1)
        else:
            break
    vc.release()
    return path + '/' + name + '/'


def img_ocr(file):
    ocr = PaddleOCR(use_angle_cls=False, lang="ch")
    result = ocr.ocr(file, cls=False)  # 全是正向数字，不需要
    return result


def pre_dealing(file, damage_type):  # ./sth/sth2.jpg
    image = cv2.imread(file)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR转HSV
    directory = os.path.split(file)[0]  # ./sth
    root_path = os.path.dirname(directory)  # .
    for i in damage_type:
        mask = cv2.inRange(hsv, element_low[i], element_high[i])
        dilate = cv2.dilate(mask, kernel)
        result = image.copy()
        result[dilate == 0] = (0, 0, 0)
        pre_dealing_image = root_path + "/pre_process/" + str(i + 1) + "_" + file.split("_")[-1]  #
        # ./pre_process/sth3.jpg
        cv2.imwrite(pre_dealing_image, result)


element_low = [(132, 82, 213), (161, 76, 185), (10, 197, 155), (0, 0, 229), (94, 140, 219), (67, 105, 218),
               (9, 59, 186), (87, 61, 237)]
element_high = [(150, 156, 255), (179, 140, 255), (20, 255, 255), (163, 12, 255), (105, 213, 255), (85, 255, 255),
                (24, 164, 255), (106, 117, 255)]
kernel = np.ones((3, 3), np.uint8)
"""
雷：(132, 82, 213), (150, 156, 255)
超载： (161, 76, 185), (179, 140, 255)
火：(10, 197, 155), (20, 255, 255)
物：(0, 0, 229), (163, 12, 255)
水：(94, 140, 219), (105, 213, 255)
风：(67, 105, 218), (85, 255, 255)
治疗：(35, 135, 113), (47, 255, 255)
岩：(9, 59, 186), (24, 164, 255)
冰：(87, 61, 237), (106, 117, 255)
"""


def batch_processing(directory, damage_type):  # ./sth/
    image = os.listdir(directory)  # sth2.jpg
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('sheet1')
    for image_name in image[::2]:  # 隔2帧取一次，不然耗时过长，并降低重复计数概率
        image_path = directory + image_name
        pre_dealing(image_path, damage_type)
    last = os.path.dirname(directory)  # ./sth
    dealt_file_path = os.path.dirname(last) + '/pre_process/'  # ./pre_process/
    dealt_image = os.listdir(dealt_file_path)
    column = ["order", "length", "height", "damage", "damage_type", "x", "y", "credibility"]
    for i, j in enumerate(column):
        worksheet.write(0, i, j)
    k = 1
    for image_name in dealt_image:
        result = img_ocr(dealt_file_path + image_name)
        location = [line[0] for line in result]
        number = [line[1][0] for line in result]
        credibility = [line[1][1] for line in result]
        damage_type = image_name[0]
        upleft_point = [line[0][0] for line in result]
        for i, j in enumerate(number):
            if number[i].isdigit():
                worksheet.write(k, 0, image_name.split("_")[-1].split(".")[0])  # 第一列几帧
                worksheet.write(k, 1, location[i][1][0] - location[i][0][0])  # 第二列框长
                worksheet.write(k, 2, location[i][2][1] - location[i][0][1])  # 第三列框高
                worksheet.write(k, 3, number[i])  # 第四列伤害(str，存在以0开头的数字,需进一步处理）
                worksheet.write(k, 4, damage_type)  # 第五列伤害类型
                worksheet.write(k, 5, upleft_point[i][0])  # 第六，七列顶点横纵坐标
                worksheet.write(k, 6, upleft_point[i][1])
                worksheet.write(k, 7, str(round(credibility[i], 3)))  # 第八列paddleocr给出的可信度
                k = k + 1
    workbook.save(dealt_file_path + "pre_process.xls")


def judge(length, height, damage):
    num = len(damage)
    return round(length / height / num, 2)


def data_dealing(file):
    df = pd.read_excel(file, dtype={'damage': str})
    df['appearance'] = df.apply(lambda x: judge(x['length'], x['height'], x['damage']), axis=1)
    unbelievable = df.groupby('damage').filter(lambda num: len(num) == 1)  # 无论识别结果是否可信，只出现一次都视为不可信
    unbelievable_list = list(zip(unbelievable.damage, unbelievable.damage_type))
    df = df.groupby('damage').filter(lambda num: len(num) > 1)
    unique = df.drop_duplicates(subset=['damage', 'damage_type'])
    most = unique[unique.credibility > 0.8].loc[unique.appearance.apply(lambda num: 0.6 <= num <= 0.8)]
    reliable = list(zip(most.damage, most.damage_type))
    uncertain = df.groupby('damage').filter(lambda num: len(num) < 3). \
        append(df.loc[df.appearance.apply(lambda num: num > 0.8 or num < 0.6)]). \
        append(df.loc[df.credibility < 0.8]).drop_duplicates()
    uncertain_unique = uncertain.drop_duplicates(subset=['damage', 'damage_type'])
    unreliable = list(zip(uncertain_unique.damage, uncertain_unique.damage_type))
    df = df[df.credibility > 0.8].groupby('damage').filter(lambda num: len(num) > 2). \
        loc[df.appearance.apply(lambda num: 0.6 <= num <= 0.8)]
    check = {}
    for i in unreliable:
        if 1 < len(i[0]) < 5:  # 对于2-4位伤害，如果头或尾两位数字和出现较多的伤害的数字相同，且伤害类型相同，视为相同伤害
            for j in reliable:
                if i[1] == j[1]:
                    if j[0].startswith(i[0][:2]) or j[0].endswith(i[0][-2:]):
                        check[i[0]] = j[0]
        elif len(i[0]) >= 5:  # 对于5位以上伤害，如果头或尾三位数字和出现较多的伤害的数字相同但不是相同伤害，且伤害类型相同，视为造成两次相同伤害
            for j in reliable:
                if i[1] == j[1] and i[0] != j[0]:
                    if i[0].startswith(j[0][:3]) or i[0].endswith(j[0][-3:]):
                        check[i[0]] = str(int(j[0]) * 2)
    uncertain.damage = uncertain.damage.apply(lambda x: check[x] if x in check.keys() else x)
    df = df.append(uncertain[uncertain.damage.isin(check.values())])
    add = {}
    for i in unbelievable_list:
        if len(i[0]) > 1:
            for j in reliable:
                if i[1] == j[1]:
                    if j[0].startswith(i[0][:2]) or j[0].endswith(i[0][-2:]):
                        add[i[0]] = j[0]
    unbelievable.damage = unbelievable.damage.apply(lambda x: add[x] if x in add.keys() else x)
    df = df.append(unbelievable[unbelievable.damage.isin(add.values())])
    too_much = df.groupby('damage').filter(lambda num: len(num) > 3)
    df = df.drop(
        too_much[(too_much.credibility < 0.9) | (too_much.appearance > 0.8) | (too_much.appearance < 0.6)].index)
    # 删除出现4次以上，可信度小于0.9或长宽比不在0.6-0.8间的数据
    df = df.loc[df.damage.apply(lambda num: 2 < len(num) < 7)]
    df.damage = df.damage.astype(dtype='int')
    types_damage = df.groupby('damage_type')['damage'].agg('sum')
    num_to_type = {1: '雷元素', 2: '超载', 3: '火元素', 4: '物理', 5: '水元素', 6: '风元素', 7: '岩元素', 8: '冰元素'}

    df_sim = df.copy()
    df_sim.order = df_sim.order.apply(lambda num: num // 18)
    df_sim.x = df_sim.x.apply(lambda num: num // 30)
    df_sim.y = df_sim.y.apply(lambda num: num // 30)
    df_sim = df_sim.drop_duplicates(subset=['order', 'x', 'y', 'damage_type', 'damage'])
    df_sim2 = df.copy()
    df_sim2.order = df_sim2.order.apply(lambda num: (num + 9) // 18)
    df_sim2.x = df_sim2.x.apply(lambda num: (num + 15) // 30)
    df_sim2.y = df_sim2.y.apply(lambda num: (num + 15) // 30)
    df_sim2 = df_sim2.drop_duplicates(subset=['order', 'x', 'y', 'damage_type', 'damage'])
    df_sim = df_sim.loc[list(set(df_sim.index).intersection(set(df_sim2.index)))]  # 别被那些无良自媒体误导了，用and取交集结果不对

    # 剩下一堆可信度贼高的重复数据，实在是不会处理了,出现7次以上就砍一半
    df_sim = df_sim[(df_sim.length < 400) & (df_sim.height < 70)]
    one_size_fits_all = list(df_sim.groupby('damage').filter(lambda num: len(num) > 7).damage.drop_duplicates())
    df_sim.damage = df_sim.damage.apply(lambda num: int(num / 2) if num in one_size_fits_all else num)

    types_damage_sim = df_sim.groupby('damage_type')['damage'].agg('sum')
    print("视频中造成的总伤害为" + str(df_sim.damage.sum()) + '-' + str(df.damage.sum()))
    for i in types_damage.index:
        print(num_to_type[i] + "伤害合计" + str(types_damage_sim[i]) + '-' + str(types_damage[i]))

    left = unbelievable[~unbelievable.damage.isin(add.values())]
    left = left.append([df[df.damage > 1000000]])
    left = left[left.credibility > 0.95]
    left = left.loc[left.damage.apply(lambda num: 2 < len(num) < 8)]
    left.damage = left.damage.astype(dtype='int')
    print('另有不能确定真实性的伤害:' + str(left.damage.sum()))  # 正常只有核爆c才需要加上该项


def main():
    root = Tk()
    root.withdraw()  # 隐藏主窗口，只弹出打开文件对话框
    video_file = filedialog.askopenfilename()  # ./sth.mp4
    input_str = input("请输入队伍可以造成以下哪些伤害，用逗号隔开\n1. 雷 2. 超载\n3. 火 4. 物理\n5. 水 6. 风\n7. 岩 8. 冰\n")
    damage_list = [int(i)-1 for i in input_str.split(',')]
    video_path = os.path.split(video_file)[0]  # .
    video_name = os.path.split(video_file)[1].split('.')[0]  # sth
    image_file = video2img(video_path, video_name, video_file)  # ./sth/
    os.makedirs(video_path + "/pre_process/")  # ./pre_process/
    batch_processing(image_file, damage_list)  # ./sth/
    data_dealing(video_path + "/pre_process/pre_process.xls")
    for root, dirs, files in os.walk(video_path + '/' + video_name):
        for name in files:
            os.remove(os.path.join(root, name))
        os.rmdir(root)
    
    for root, dirs, files in os.walk(video_path + '/pre_process'):
    for name in files:
        os.remove(os.path.join(root, name))
    os.rmdir(root)
    


if __name__ == '__main__':
    main()
