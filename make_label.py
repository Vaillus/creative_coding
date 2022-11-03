import numpy as np
from PIL import Image
import os

def get_centroids(file_name):
    """get the txt file outputed by yolov5 and get the 128x128 coordinates
    of the centroids of the leopard spots.
    """
    lines = []
    with open(file_name) as f:
        lines = f.readlines()
        points = []
    for line in lines:
        line = line.split()
        if len(line) > 0:
            line.pop(0)
            line = np.array(line).astype(np.float) * 128
            line = np.round(line)
            point = (round(line[0]), round(line[1]))
            points += [point]
    return points

def get_label(points):
    label = np.zeros(shape=(128, 128))
    for point in points:
        label[point] = 1
    return label

def make_labelled_img(img_id):
    points = get_centroids("./labels/"+img_id+".txt")
    img = Image.open("in/128/" + img_id + ".png")
    pix = img.load()
    for point in points:
        pix[point[0],point[1]] = (255, 0, 0)
    img.save("./out/test_label/" + img_id + "-128.png")

def resize_images(folder, size):
    for img_name in os.listdir(folder):
        img = Image.open(folder + img_name)
        img = img.resize((size, size))
        img.save("in/" + str(size) + "/" + img_name)

if __name__ == '__main__':
    resize_images("/mnt/c/Users/vaill/Pictures/léopard-cropped/", 512)
    #make_labelled_img("44-1")
    # points = get_centroids(file_name)
    # label = get_label(points)
    
