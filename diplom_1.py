#поиск ведется на основе границ кенни 
import numpy as np
import cv2 as cv
import tkinter.filedialog as fd
import math
import imutils
def viewImage(image, name_of_window):
    cv.namedWindow(name_of_window, cv.WINDOW_NORMAL)
    cv.imshow(name_of_window, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
# параметры цветового фильтра
hsv_min = np.array((59, 119, 17), np.uint8)
hsv_max = np.array((79, 255, 255), np.uint8)
filename = fd.askopenfilename(title="Открыть шаблон")
img = cv.imread(filename)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV )
# меняем цветовую модель с BGR на HSV
thresh = cv.inRange(hsv, hsv_min, hsv_max )
# применяем цветовой фильтр
# ищем контуры и складируем их в переменную contours
contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

maxX = 0
minX = 999999
maxY = 0
minY= 999999
for i in contours:
    if( maxX < i[0][0][0]):
        maxX = i[0][0][0]
    if( maxY < i[0][0][1]):
        maxY = i[0][0][1]
    if( minX > i[0][0][0]):
        minX = i[0][0][0]
    if( minY > i[0][0][1]):
        minY = i[0][0][1]

image12 = cv.imread(filename)
cv.rectangle(image12, (minX, minY), (maxX, maxY), (0, 255, 255), 10)
cropped = image12[minY:maxY, minX:maxX]
hsv_min = np.array((0, 54, 5), np.uint8)
hsv_max = np.array((187, 255, 253), np.uint8)


if __name__ == '__main__':
    img = cropped

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
    thresh = cv.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # перебираем все найденные контуры в цикле
    for cnt in contours0:

        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        center = (int(rect[0][0]), int(rect[0][1]))

        # вычисление координат двух векторов, являющихся сторонам прямоугольника
        edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
        edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))
        # выясняем какой вектор больше
        reference = (1, 0)  # горизонтальный вектор, задающий горизонт
        # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
        angle1 = 180.0 / math.pi * math.acos((reference[0] * edge1[0] + reference[1] * edge1[1]) / (cv.norm(reference) * cv.norm(edge1)))
        angle2 = 180.0 / math.pi * math.acos((reference[0] * edge2[0] + reference[1] * edge2[1]) / (cv.norm(reference) * cv.norm(edge2)))
        x1 = box[0][0]
        y1 = box[0][1]
        x2 = box[1][0]
        y2 = box[1][1]
        x3 = box[2][0]
        y3 = box[2][1]
        x4 = box[3][0]
        y4 = box[3][1]

        S =float(1/2*(x1*y2+x2*y3+x3*y4+x4*y1-x2*y1-x3*y2-x4*y3-x1*y4))
        if(S>=300 and S<25000 and (angle1<15 or angle2<15)):
            l1 = float((abs(x1 - x2) + abs(y1 - y2)) ** (1 / 2))
            l2 = float((abs(x3 - x2) + abs(y3 - y2)) ** (1 / 2))
            print(float(l1 / l2))
            print(float(l2 / l1))
            print()
            if(abs(l1-l2)<3):
                cv.drawContours(img, [box], 0, (255, 255, 255), 2)  # рисуем прямоугольник

    cv.imshow('contours', imutils.resize( img, width = 1600))  # вывод обработанного кадра в окно

    cv.waitKey()
    cv.destroyAllWindows()

