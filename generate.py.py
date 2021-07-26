#!/usr/bin/env python

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import copy
import os.path
import time

import carla

from carla import ColorConverter as cc

import cv2
import re

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# Create Directory ################
dir_my = 'my_data/'
dir_custom = 'custom_data/'
dir_draw = 'draw_bounding_box/'
if not os.path.exists(dir_my):
    os.makedirs(dir_my)
if not os.path.exists(dir_custom):
    os.makedirs(dir_custom)
if not os.path.exists(dir_draw):
    os.makedirs(dir_draw)
###################################

dataEA = len(os.walk('VehicleBBox/').next()[2])

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

VBB_COLOR = (0, 0, 255)
WBB_COLOR = (255, 0, 0)

Vehicle_COLOR = np.array([142, 0, 0])
Walker_COLOR = np.array([60, 20, 220])

rgb_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
seg_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)
index_count = 0

# Brings Images and Bounding Box Information
def reading_data(index):
    global rgb_info, seg_info
    v_data = []
    w_data = []
    k = 0
    w = 0

    rgb_img = cv2.imread('custom_data/image'+ str(index)+ '.png', cv2.IMREAD_COLOR)
    seg_img = cv2.imread('SegmentationImage/seg'+ str(index)+ '.png', cv2.IMREAD_COLOR)

    if str(rgb_img) != "None" and str(seg_img) != "None":
        # Vehicle
        with open('VehicleBBox/bbox'+ str(index), 'r') as fin:
            v_bounding_box_rawdata = fin.read()

        v_bounding_box_data = re.findall(r"-?\d+", v_bounding_box_rawdata)
        v_line_length = len(v_bounding_box_data) / 16 

        v_bbox_data = [[0 for col in range(8)] for row in range(v_line_length)] 

        for i in range(len(v_bounding_box_data)/2):
            j = i*2
            v_data.append(tuple((int(v_bounding_box_data[j]), int(v_bounding_box_data[j+1]))))

        for i in range(len(v_bounding_box_data)/16):
            for j in range(8):
                v_bbox_data[i][j] = v_data[k]
                k += 1

        # Walker (Pedestrian)
        with open('PedestrianBBox/bbox'+ str(index), 'r') as w_fin:
            w_bounding_box_rawdata = w_fin.read()

        w_bounding_box_data = re.findall(r"-?\d+", w_bounding_box_rawdata)
        w_line_length = len(w_bounding_box_data) / 16 

        w_bb_data = [[0 for col in range(8)] for row in range(w_line_length)] 

        for i in range(len(w_bounding_box_data)/2):
            j = i*2
            w_data.append(tuple((int(w_bounding_box_data[j]), int(w_bounding_box_data[j+1]))))

        for i in range(len(w_bounding_box_data)/16):
            for j in range(8):
                w_bb_data[i][j] = w_data[w]
                w += 1

        origin_rgb_info = rgb_img
        rgb_info = rgb_img
        seg_info = seg_img
        return v_bbox_data, v_line_length, w_bb_data, w_line_length 

    else:
        return False

# Converts 8 Vertices to 4 Vertices
def converting(bounding_boxes, line_length):
    points_array = []
    bb_4data = [[0 for col in range(4)] for row in range(line_length)]
    k = 0
    for i in range(line_length):
        points_array_x = []
        points_array_y = []      
        for j in range(8):
            points_array_x.append(bounding_boxes[i][j][0])
            points_array_y.append(bounding_boxes[i][j][1])

            max_x = max(points_array_x)
            min_x = min(points_array_x)
            max_y = max(points_array_y)
            min_y = min(points_array_y)           

        points_array.append(tuple((min_x, min_y)))
        points_array.append(tuple((max_x, min_y)))
        points_array.append(tuple((max_x, max_y)))
        points_array.append(tuple((min_x, max_y)))

    for i in range(line_length):
        for j in range(len(points_array)/line_length):
            bb_4data[i][j] = points_array[k]
            k += 1  

    return bb_4data

# Gets Object's Bounding Box Area
def object_area(data):
    global area_info
    area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)

    for vehicle_area in data:
        array_x = []
        array_y = []
        for i in range(4):
           array_x.append(vehicle_area[i][0])
        for j in range(4):
           array_y.append(vehicle_area[j][1])

        for i in range(4):
            if array_x[i] <= 0:
                array_x[i] = 1
            elif array_x[i] >= VIEW_WIDTH:
                array_x[i] = VIEW_WIDTH -1
        for j in range(4):
            if array_y[j] <= 0:
                array_y[j] = 1
            elif array_y[j] >= VIEW_HEIGHT:
                array_y[j] = VIEW_HEIGHT -1
       
        min_x = min(array_x) 
        max_x = max(array_x) 
        min_y = min(array_y) 
        max_y = max(array_y) 
        array = [min_x, max_x, min_y, max_y]
        if filtering(array, Vehicle_COLOR): 
            cv2.rectangle(area_info, (min_x, min_y), (max_x, max_y), Vehicle_COLOR, -1)

# Fits Bounding Box to the Object
def fitting_x(x1, x2, range_min, range_max, color):
    global seg_info
    state = False
    cali_point = 0
    if (x1 < x2):
        for search_point in range(x1, x2):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(x1, x2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    return cali_point

def fitting_y(y1, y2, range_min, range_max, color):
    global seg_info
    state = False
    cali_point = 0
    if (y1 < y2):
        for search_point in range(y1, y2):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(y1, y2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    return cali_point

# Removes small objects that obstruct to learning
def small_objects_excluded(array, bb_min):
    diff_x = array[1]- array[0]
    diff_y = array[3] - array[2]
    if (diff_x > bb_min and diff_y > bb_min):
        return True
    return False

# Filters occluded objects
def post_occluded_objects_excluded(array, color):
    global seg_info
    top_left = seg_info[array[2]+1, array[0]+1][0]
    top_right = seg_info[array[2]+1, array[1]-1][0] 
    bottom_left = seg_info[array[3]-1, array[0]+1][0] 
    bottom_right = seg_info[array[3]-1, array[1]-1][0]
    if top_left == color[0] and top_right == color[0] and bottom_left == color[0] and bottom_right == color[0]:
        return False

    return True

def pre_occluded_objects_excluded(array, area_image, color):
    top_left = area_image[array[2]-1, array[0]-1][0]
    top_right = area_image[array[2], array[1]+1][0] 
    bottom_left = area_image[array[3]+1, array[1]-1][0] 
    bottom_right = area_image[array[3]+1, array[0]+1][0]
    if top_left == color[0] and top_right == color[0] and bottom_left == color[0] and bottom_right == color[0]:
        return False

    return True

# Filters objects not in the scene
def filtering(array, color):
    global seg_info
    for y in range(array[2], array[3]):
        for x in range(array[0], array[1]):
            if seg_info[y, x][0] == color[0]:
                return True
    return False

# Processes Post-Processing
def processing(img, v_data, w_data, index):
    global seg_info, area_info
    vehicle_class = 0
    walker_class = 1

    object_area(v_data)
    f = open("custom_data/image"+str(index) + ".txt", 'w')

    # Vehicle
    for v_bbox in v_data:
        array_x = []
        array_y = []
        for i in range(4):
           array_x.append(v_bbox[i][0])
        for j in range(4):
           array_y.append(v_bbox[j][1])

        for i in range(4):
            if array_x[i] <= 0:
                array_x[i] = 1
            elif array_x[i] >= VIEW_WIDTH - 1:
                array_x[i] = VIEW_WIDTH - 2
        for j in range(4):
            if array_y[j] <= 0:
                array_y[j] = 1
            elif array_y[j] >= VIEW_HEIGHT - 1:
                array_y[j] = VIEW_HEIGHT - 2
       
        min_x = min(array_x) 
        max_x = max(array_x) 
        min_y = min(array_y) 
        max_y = max(array_y) 
        v_bb_array = [min_x, max_x, min_y, max_y]
        center_x = (min_x + max_x)//2
        center_y = (min_y + max_y)//2

        if filtering(v_bb_array, Vehicle_COLOR) and pre_occluded_objects_excluded(v_bb_array, area_info, Vehicle_COLOR): 
            cali_min_x = fitting_x(min_x, max_x, min_y, max_y, Vehicle_COLOR)
            cali_max_x = fitting_x(max_x, min_x, min_y, max_y, Vehicle_COLOR)
            cali_min_y = fitting_y(min_y, max_y, min_x, max_x, Vehicle_COLOR)
            cali_max_y = fitting_y(max_y, min_y, min_x, max_x, Vehicle_COLOR)
            v_cali_array = [cali_min_x, cali_max_x, cali_min_y, cali_max_y]

            if small_objects_excluded(v_cali_array, 10) and post_occluded_objects_excluded(v_cali_array, Vehicle_COLOR):
                darknet_x = float((cali_min_x + cali_max_x) // 2) / float(VIEW_WIDTH)
                darknet_y = float((cali_min_y + cali_max_y) // 2) / float(VIEW_HEIGHT)
                darknet_width = float(cali_max_x - cali_min_x) / float(VIEW_WIDTH)
                darknet_height= float(cali_max_y - cali_min_y) / float(VIEW_HEIGHT)

                f.write(str(vehicle_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
                str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

                cv2.line(img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), VBB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), VBB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), VBB_COLOR, 2)
                cv2.line(img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), VBB_COLOR, 2)

    # Walker (Pedestrian)
    object_area(w_data)

    for wbbox in w_data:
        array_x = []
        array_y = []

        for i in range(4):
           array_x.append(wbbox[i][0])
        for j in range(4):
           array_y.append(wbbox[j][1])

        for i in range(4):
            if array_x[i] <= 0:
                array_x[i] = 1
            elif array_x[i] >= VIEW_WIDTH - 1:
                array_x[i] = VIEW_WIDTH - 2
        for j in range(4):
            if array_y[j] <= 0:
                array_y[j] = 1
            elif array_y[j] >= VIEW_HEIGHT - 1:
                array_y[j] = VIEW_HEIGHT - 2
       
        min_x = min(array_x) 
        max_x = max(array_x) 
        min_y = min(array_y) 
        max_y = max(array_y)
        w_bb_array = [min_x, max_x, min_y, max_y]
        if filtering(w_bb_array, Walker_COLOR) and pre_occluded_objects_excluded(w_bb_array, area_info, Walker_COLOR): 
            cali_min_x = fitting_x(min_x, max_x, min_y, max_y, Walker_COLOR)
            cali_max_x = fitting_x(max_x, min_x, min_y, max_y, Walker_COLOR)
            cali_min_y = fitting_y(min_y, max_y, min_x, max_x, Walker_COLOR)
            cali_max_y = fitting_y(max_y, min_y, min_x, max_x, Walker_COLOR)
            w_cali_array = [cali_min_x, cali_max_x, cali_min_y, cali_max_y]

            if small_objects_excluded(w_cali_array, 7) and post_occluded_objects_excluded(w_cali_array, Walker_COLOR):
                darknet_x = float((cali_min_x + cali_max_x) // 2) / float(VIEW_WIDTH)
                darknet_y = float((cali_min_y + cali_max_y) // 2) / float(VIEW_HEIGHT)
                darknet_width = float(cali_max_x - cali_min_x) / float(VIEW_WIDTH)
                darknet_height= float(cali_max_y - cali_min_y) / float(VIEW_HEIGHT)

                f.write(str(walker_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
                str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

                cv2.line(img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), WBB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), WBB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), WBB_COLOR, 2)
                cv2.line(img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), WBB_COLOR, 2)

    f.close()
    cv2.imwrite('draw_bounding_box/image'+str(index)+'.png', img)

def run():
    global rgb_info
    global index_count
    train = open("my_data/train.txt", 'w')

    for i in range(dataEA + 1):
        if reading_data(i) != False:
            v_four_points = converting(reading_data(i)[0], reading_data(i)[1])
            w_four_points = converting(reading_data(i)[2], reading_data(i)[3])
            processing(rgb_info, v_four_points, w_four_points, i)
            train.write(str('custom_data/image'+str(i) + '.png') + "\n")
            index_count = index_count + 1
            print(i)
    train.close()
    print(index_count)

if __name__ == "__main__":
    start = time.time()

    run()

    end = time.time()
    print(float(end - start))
