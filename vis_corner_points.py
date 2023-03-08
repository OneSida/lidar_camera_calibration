import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt


points_txt_file = './conf/points.txt'
points_list = []
with open('./conf/points.txt') as fin:
    num_points = int(fin.readline().strip())
    for line in fin:
        line = line.strip().split(' ')
        points_list.append([float(item) for item in line])
half_num_points = int(num_points / 2)
lidar_points = points_list[:num_points]
# pprint(lidar_points)
camera_points = points_list[num_points:]
# pprint(camera_points)

final_T = None
with open('./log/avg_values.txt') as fin:
    while True:
        line = fin.readline().strip()
        if not line:
            break
        iter_id = int(line)
        print('iter_id:', iter_id)
        dx = fin.readline().strip()
        dy = fin.readline().strip()
        dz = fin.readline().strip()
        T = []
        T.append([float(item) for item in (fin.readline().strip() + ' ' + dx).split(' ')])
        T.append([float(item) for item in (fin.readline().strip() + ' ' + dy).split(' ')])
        T.append([float(item) for item in (fin.readline().strip() + ' ' + dz).split(' ')])
        T.append([0, 0, 0, 1])
        final_T = T
        print('rmse_raw:', fin.readline().strip())
# pprint(final_T)

homo_lidar_points = np.concatenate((np.array(lidar_points).T, \
                                    np.ones((1, len(lidar_points)))))
# pprint(homo_lidar_points)
transformed_homo_lidar_points = np.dot(final_T, homo_lidar_points)
transformed_lidar_points = transformed_homo_lidar_points[:-1, :].T
print(transformed_lidar_points.shape)
camera_points = np.array(camera_points)
print(camera_points.shape)

rmse = np.sqrt(np.sum(np.power(transformed_lidar_points - camera_points, 2)) / num_points / 3)
print('rmse:', rmse)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(transformed_lidar_points[:half_num_points, 0], \
            transformed_lidar_points[:half_num_points, 1], \
            transformed_lidar_points[:half_num_points, 2], 'red')
ax.scatter3D(transformed_lidar_points[half_num_points:, 0], \
            transformed_lidar_points[half_num_points:, 1], \
            transformed_lidar_points[half_num_points:, 2], 'yellow')
ax.scatter3D(camera_points[:half_num_points, 0], \
            camera_points[:half_num_points, 1], \
            camera_points[:half_num_points, 2], 'green')
ax.scatter3D(camera_points[half_num_points:, 0], \
            camera_points[half_num_points:, 1], \
            camera_points[half_num_points:, 2], 'blue')
plt.show()


