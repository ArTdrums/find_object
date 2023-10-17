import time
import math
import cv2
from config import *
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

start = time.perf_counter()
if __name__ == '__main__':
    def nothing(*arg):
        pass

cv2.namedWindow("result")  # создаем главное окно
cv2.namedWindow("settings")  # создаем окно настроек

# создаем 7 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('hue_1', 'settings', 0, 180, nothing)
cv2.createTrackbar('saturation_1', 'settings', 0, 255, nothing)
cv2.createTrackbar('value_1', 'settings', 0, 255, nothing)
cv2.createTrackbar('hue_2', 'settings', 255, 255, nothing)
cv2.createTrackbar('saturation_2', 'settings', 255, 255, nothing)
cv2.createTrackbar('value_2', 'settings', 255, 255, nothing)
cv2.createTrackbar('blurring', 'settings', 1, 25, nothing)  # , бегунок для размытия

crange = [0, 0, 0, 0, 0, 0]

while True:

    img = cv2.imread('images/image_1.jpg')
    blurring = cv2.getTrackbarPos('blurring', 'settings')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    def blurring_check(blurring: int) -> int:
        global img

        # проверка числа на четность

        if blurring % 2 != 0:
            img = cv2.GaussianBlur(img, (blurring, 15), 0)  #

        else:  # если не четное, увеличиваем значение q1 на 1
            blurring += 1
            img = cv2.GaussianBlur(img, (blurring, 15), 0)
            return blurring


    t1 = Thread(target=blurring_check, args=int(blurring))

    # считываем значения бегунков
    h1 = cv2.getTrackbarPos('hue_1', 'settings')
    s1 = cv2.getTrackbarPos('saturation_1', 'settings')
    v1 = cv2.getTrackbarPos('value_1', 'settings')
    h2 = cv2.getTrackbarPos('hue_2', 'settings')
    s2 = cv2.getTrackbarPos('saturation_2', 'settings')
    v2 = cv2.getTrackbarPos('value_2', 'settings')
    q1 = cv2.getTrackbarPos('blurring', 'settings')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # формируем начальный и конечный цвет фильтра
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    # накладываем фильтр на кадр в модели HSV
    filter = cv2.inRange(hsv, h_min, h_max)

    # наклажываем маску на оригинальное изображение
    orig_image_with_mask = cv2.bitwise_and(img, img, mask=filter)
    # делаем выводы маски и наложение маски на оригинальное  изображение

    cv2.imshow('window_name', orig_image_with_mask)

    cv2.imshow('result', filter)

    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()
# загрузка изображения
#img = cv2.imread('images/image_1.jpg')



time.sleep(0.5)
# вывод отфильтрованного изображения на экран
# cv2.imshow('color_hsv',  filter)

cv2.waitKey(1000)
# находим контуры
contours, poisk = cv2.findContours(filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

time.sleep(0.5)

for icontour in tqdm(contours, desc='процесс'):
    # ищем прямоугольник, результат записываем в rect
    rect = cv2.minAreaRect(icontour)
    # поиск вершин прямоугольника, результат записываем
    # в box
    area = int(rect[1][0] * rect[1][1])

    if area > param_square_low and area < param_square_hight:
        # поиск вершин прямоугольника, результат записываем
        # в box

        box = cv2.boxPoints(rect)
        # округление координат вершин, результат
        # записываем в box

        box = np.int0(box)

        vec1 = np.int0((box[1][0] - box[0][0],  # рассчитываем координаты первого вектора
                        box[1][1] - box[0][1]))

        vec2 = np.int0((box[2][0] - box[1][0],  # рассчитываем координаты второго вектора
                        box[2][1] - box[1][1]))

        used_vec = vec1  # я не знаю какой вектор будет больше  , поэтому


        # предполагаю , что это будет первый вектор,
        # сохраняем его в used_vec

        def angle_check(vec2: int, vec1: int) -> np.array:
            global center
            global angle
            global used_vec

            if cv2.norm(vec2) > cv2.norm(vec1):  # если длина второго вектора больше первого,
                # значит, в used_vec сохраним длину второго
                # вектора

                used_vec = vec2

            center = (int(rect[0][0]), int(rect[0][1]))  # записываем координаты центра прямоугольника

            angle = 180.0 / math.pi * math.acos(used_vec[0] /
                                                cv2.norm(used_vec))  # рассчитываем угол наклона прямоугольника


        threading.Thread(target=angle_check(vec2, vec1)).start()


        def square_check(angle: int) -> np.array:

            if int(angle) >= square_count:  # проверка на угол поворота обьета относительно оси Х

                cv2.drawContours(img, [box], -1,  # рисуем прямоугольник
                                 red, 3)

                cv2.circle(img, center, 5, green, 2)  # рисуем окружность в центре
                # прямоугольника

                cv2.putText(img, "%d" % int(angle),  # выводим рядом с прямоугольником значение угла
                            # наклона

                            (center[0] + 10, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
            else:
                pass


        threading.Thread(target=square_check(angle_count)).start()

plt.imshow(img)
time.sleep(0.5)
plt.show()

print(time.perf_counter() - start)
