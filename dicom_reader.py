import os
import math
import pydicom
from matplotlib import pyplot as plt
import numpy as np

def distance_squared(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    dx = x1 - x2
    dy = y1 - y2

    return dx * dx + dy * dy


def closest_point(all_points, new_point):
    best_point = None
    best_distance = None

    for current_point in all_points:
        current_distance = distance_squared(new_point, current_point)

        if best_distance is None or current_distance < best_distance:
            best_distance = current_distance
            best_point = current_point

    return best_point


k = 2


def build_kdtree(points, depth=0):
    n = len(points)

    if n <= 0:
        return None

    axis = depth % k

    sorted_points = sorted(points, key=lambda point: point[axis])

    return {
        'point': sorted_points[n // 2],
        'left': build_kdtree(sorted_points[:n // 2], depth + 1),
        'right': build_kdtree(sorted_points[n // 2 + 1:], depth + 1)
    }


def kdtree_naive_closest_point(root, point, depth=0, best=None):
    if root is None:
        return best

    axis = depth % k

    next_best = None
    next_branch = None

    if best is None or distance_squared(point, best) > distance_squared(point, root['point']):
        next_best = root['point']
    else:
        next_best = best

    if point[axis] < root['point'][axis]:
        next_branch = root['left']
    else:
        next_branch = root['right']

    return kdtree_naive_closest_point(next_branch, point, depth + 1, next_best)


def closer_distance(pivot, p1, p2):
    if p1 is None:
        return p2

    if p2 is None:
        return p1

    d1 = distance_squared(pivot, p1)
    d2 = distance_squared(pivot, p2)

    if d1 < d2:
        return p1
    else:
        return p2


def kdtree_closest_point(root, point, depth=0):
    if root is None:
        return None

    axis = depth % k

    next_branch = None
    opposite_branch = None

    if point[axis] < root['point'][axis]:
        next_branch = root['left']
        opposite_branch = root['right']
    else:
        next_branch = root['right']
        opposite_branch = root['left']

    best = closer_distance(point,
                           kdtree_closest_point(next_branch,
                                                point,
                                                depth + 1),
                           root['point'])

    if distance_squared(point, best) > (point[axis] - root['point'][axis]) ** 2:
        best = closer_distance(point,
                               kdtree_closest_point(opposite_branch,
                                                    point,
                                                    depth + 1),
                               best)

    return best

# paths to data and save location
filepath = './data/' # directory containing the dicom series
dcmprefix = 'I' # Individual dicom file prefix before number
firstdcm = dcmprefix + '12' # first dicom image to inspect metadata fields
newdir = './data-edited/' # directory to save new metadata fields

# Find total number of dicom files in series
j = 0
for file in os.listdir(filepath):
    if file.endswith(".DICOM"):
        j = j + 1
totaldcm = j 

# Read all tags for first image
ds = pydicom.filereader.dcmread(filepath+firstdcm)
print(ds.pixel_array.shape)
print(ds.pixel_array[200][130])
imageArray = ds.pixel_array

# mask = np.identity(512)
plt.imshow(imageArray, cmap=plt.cm.bone)  # set the color map to bone
plt.show()

mask = np.zeros((512,512))
result = np.zeros((512,512))

myPoints = []
for x in range(len(imageArray)):
    for y in range(len(imageArray[x])):
        if imageArray[x,y] > 20 and imageArray[x,y] < 300:
            mask[x,y]= 1
            myPoints.append((x,y))
plt.imshow(mask, cmap='gray')  # set the color map to bone
plt.show()

pivot = (256,256)

kdtree = build_kdtree(myPoints)
found = kdtree_closest_point(kdtree, pivot)
found_distance = math.sqrt(distance_squared(pivot, found))
print("  Found:    %s (distance: %f)" % (found, found_distance))
result[found[0],found[1]] = 1

while found_distance < 35:
    imageArray[found[0],found[1]] = 0
    myPoints = []
    for x in range(len(imageArray)):
        for y in range(len(imageArray[x])):
            if imageArray[x,y] > 20 and imageArray[x,y] < 300:
                myPoints.append((x,y))
    kdtree = build_kdtree(myPoints)
    found = kdtree_closest_point(kdtree, pivot)
    print(found)
    found_distance = math.sqrt(distance_squared(pivot, found))
    # print("  Found:    %s (distance: %f)" % (found, found_distance))
    result[found[0],found[1]] = 1

plt.imshow(result, cmap='gray')  # set the color map to bone
plt.show()
