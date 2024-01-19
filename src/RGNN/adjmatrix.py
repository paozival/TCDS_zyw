import numpy as np
import math as m
from matplotlib import pyplot as plt
from einops import rearrange

# 本程序用于计算GNN中所用的脑电邻接矩阵A 初始化方法来自于文献
# ”EEG-Based Emotion Recognition Using Regularized Graph Neural Networks“
# 其中包括local connection和global connection
# Symmetrical, contains self-loops
# Calibration constant should keep 20% of the links according to the paper
def get_adjacency_matrix(channel_names, positions_3d, global_connections, calibration_constant=5, active_threshold=0.1):
    # 易证以下程序结果=get_3d_distance()
    distance_3d_matrix = np.array([positions_3d, ] * len(positions_3d))
    # Transpose
    distance_3d_matrix = rearrange(distance_3d_matrix, 'h w d -> w h d')
    # Calculate 3d distances (m.sqrt(incX**2 + incY**2 + incZ**2))
    distance_3d_matrix = distance_3d_matrix - positions_3d
    distance_3d_matrix = distance_3d_matrix ** 2
    distance_3d_matrix = distance_3d_matrix.sum(axis=-1)
    distance_3d_matrix = np.sqrt(distance_3d_matrix)
    # Define local connections aij=min(a,δ/dij) 原论文是dij**2感觉有点问题
    # δ=5 大约20%的local connections不可忽略,规定aij>0.1的connections被视作不可忽略
    distance_3d_matrix = calibration_constant / distance_3d_matrix
    distance_3d_matrix[distance_3d_matrix > 1 ] = 1

    local_conn_mask = distance_3d_matrix > active_threshold
    local_conn_matrix = distance_3d_matrix * local_conn_mask
    # Min max normalize connections and initialize adjacency_matrix
    np.fill_diagonal(local_conn_matrix, 0)
    # adj_matrix = (local_connections - local_connections.min()) / (local_connections.max() - local_connections.min())

    # Global connections get initialized to aij-1
    global_connections = [[np.where(channel_names == e[0])[0], np.where(channel_names == e[1])[0]] for e in global_connections]
    global_connections = np.array(global_connections).squeeze()

    # Set symmetric global connections
    global_conn_matrix = local_conn_matrix
    global_conn_matrix[global_connections[:, 0], global_connections[:, 1]] = local_conn_matrix[global_connections[:, 0], global_connections[:, 1]]-1
    global_conn_matrix[global_connections[:, 1], global_connections[:, 0]] = local_conn_matrix[global_connections[:, 1], global_connections[:, 0]]-1
    adj_matrix = global_conn_matrix
    return adj_matrix


r""" 
距离投影函数汇总
# 计算距离有两种方法：1、将三维坐标投影至二维平面后计算二维距离 2、直接进行三维距离的计算
# Helper function for get_projected_2d_positions
def azim_proj(self, pos):
    r, elev, az = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


# Helper function for get_projected_2d_positions 直角坐标转换为球坐标
def cart2sph(x, y, z):
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt(x2_y2 + z ** 2)  # r     tant^(-1)(y/x)
    elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation仰角
    az = m.atan2(y, x)  # Azimuth方位角
    return r, elev, az


# Helper function for get_projected_2d_positions 极坐标转换为直角坐标
def pol2cart(theta, rho):
    return rho * m.cos(theta), rho * m.sin(theta)


def get_projected_2d_positions(positions_3d):
    # projected投影获得二位坐标
    pos_2d = np.array([azim_proj(pos_3d) for pos_3d in positions_3d])
    return pos_2d


# Distance using projected coordinates 投影计算二维距离
def get_projected_2d_distance(channel_names, name1, name2):
    index1, index2 = np.where(channel_names == name1)[0][0], np.where(channel_names == name2)[0][0]
    p1, p2 = get_projected_2d_positions[index1], get_projected_2d_positions[index2]
    incX, incY = p1[0] - p2[0], p1[1] - p2[1]
    return m.sqrt(incX ** 2 + incY ** 2)

# 2d projection visualization
def plot_2d_projection(channel_names):
    fig, ax = plt.subplots()
    ax.scatter(get_projected_2d_positions[:, 0], get_projected_2d_positions[:, 1])
    for i, name in enumerate(channel_names):
        plt.text(get_projected_2d_positions[:, 0][i], get_projected_2d_positions[:, 1][i], name)
    plt.show()


# 直接运用三维坐标 Distance using 3d positions
def get_3d_distance(channel_names,positions_3d, name1, name2):
    index1, index2 = np.where(channel_names == name1)[0][0], np.where(channel_names == name2)[0][0]
    p1, p2 = positions_3d[index1], positions_3d[index2]
    incX, incY, incZ = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
    return m.sqrt(incX ** 2 + incY ** 2 + incZ ** 2)
"""