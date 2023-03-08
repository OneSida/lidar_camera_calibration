import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from pprint import pprint
import cv2


pcd_file = '../../data/pcd/1508951983188262.pcd'
jpg_file = '../../data/jpg/frame0010.jpg'
image_w = 640
image_h = 480
camera_matrix = np.array([
    [595.430298, 0., 334.968617],
    [0., 598.599609, 228.841538],
    [0., 0., 1.]
    # [599.425037, 0.000000, 331.926573],
    # [0.000000, 597.969408, 228.704189],
    # [0.000000, 0.000000, 1.000000]
])
dist_coeffs = np.array([0.029251, -0.169672, 0.000825, 0.004710, 0.])


def project():
    final_T = np.zeros((4, 4))
    final_T[3, 3] = 1
    with open('./calib_result.txt') as fin:
        fin.readline()
        for i in range(4):
            final_T[i, 3] = float(fin.readline().strip().split(' ')[-1])
        fin.readline()
        for i in range(4):
            j = 0
            for item in fin.readline().strip().split(' '):
                if not item:
                    continue
                final_T[i, j] = float(item)
                j += 1
    # with open('./log/avg_values.txt') as fin:
    #     while True:
    #         line = fin.readline().strip()
    #         if not line:
    #             break
    #         iter_id = int(line)
    #         # print('iter_id:', iter_id)
    #         dx = fin.readline().strip()
    #         dy = fin.readline().strip()
    #         dz = fin.readline().strip()
    #         T = []
    #         T.append([float(item) for item in (fin.readline().strip() + ' ' + dx).split(' ')])
    #         T.append([float(item) for item in (fin.readline().strip() + ' ' + dy).split(' ')])
    #         T.append([float(item) for item in (fin.readline().strip() + ' ' + dz).split(' ')])
    #         T.append([0, 0, 0, 1])
    #         final_T = np.array(T)
    #         rmse_raw = fin.readline().strip()
            # print('rmse_raw:', rmse_raw)
    # print(final_T)
    # pprint(final_T)
    # final_T = np.array([
    #     [0.996554, -0.00467748, -0.0828156, 0.0894906],
    #     [0.0109373, 0.997101, 0.0752965, 0.156528],
    #     [0.0822233, -0.0759428, 0.993716, 0.388588],
    #     [0, 0, 0, 1]
    # ])
    # final_T[:3, :3] = np.array([
    #     [-0.045864, -0.9989, 0.00974143],
    #     [0.0252594, -0.0109082, -0.999621],
    #     [0.998628, -0.0456005, 0.025732]
    # ])
    
    rot_mat = final_T[:3, :3]
    rvec = cv2.Rodrigues(rot_mat)[0].T[0]
    tvec = final_T[:3, 3]
    # print(rvec, tvec)

    pcd = o3d.io.read_point_cloud(pcd_file)
    # o3d.visualization.draw_geometries([pcd])
    points = np.array(pcd.points)
    points = points[points[:, 0] > 0.]
    points = points[points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2 < 2.5**2]
    # pprint(points)
    # homo_lidar_points = np.concatenate((np.array(points).T, \
    #                                 np.ones((1, len(points)))))
    # transformed_homo_lidar_points = np.dot(final_T, homo_lidar_points)
    # points = transformed_homo_lidar_points[:-1, :].T
    # points = points[points[:, 2] > 0.]
    # pprint(points)
    # points = np.array([[0., 0., 2.]])
    # rvec = np.zeros(3)
    # tvec = np.zeros(3)
    points_2d, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    # pprint(points_2d)
    # return
    x_list = []
    y_list = []
    for point in points_2d:
        x = point[0][0]
        y = point[0][1]
        if 0 < x < image_w and 0 < y < image_h:
            x_list.append(x)
            y_list.append(y)
        # print(x, y)
        # break
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    # pprint(x_list)
    # pprint(y_list)
    jpg = cv2.imread(jpg_file)
    plt.imshow(jpg)
    plt.scatter(x_list, y_list, marker='.', color='red', alpha=0.1)
    plt.show()


if __name__ == '__main__':
    project()
    pass

