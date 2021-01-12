'''
Creating functions in this script to use in starter_code.py
'''
import numpy as np
import math
import os

def rotation_matrix_x(theta):
    C = np.zeros((3, 3))
    theta_rad = theta * math.pi / 180
    C[0,0] = 1
    C[1,1] = math.cos(theta_rad)
    C[1,2] = -math.sin(theta_rad)
    C[2,1] = math.sin(theta_rad)
    C[2,2] = math.cos(theta_rad)
    return C

def rotation_matrix_y(theta):
    C = np.zeros((3, 3))
    theta_rad = theta * math.pi / 180
    C[0,0] = math.cos(theta_rad)
    C[0,2] = math.sin(theta_rad)
    C[1,1] = 1
    C[2,0] = -math.sin(theta_rad)
    C[2,2] = math.cos(theta_rad)
    return C

def rotation_matrix_z(theta):
    C = np.zeros((3, 3))
    theta_rad = theta * math.pi / 180
    C[0,0] = math.cos(theta_rad)
    C[0,1] = -math.sin(theta_rad)
    C[1,0] = math.sin(theta_rad)
    C[1,1] = math.cos(theta_rad)
    C[2,2] = 1
    return C

def rotation_matrix_tait_bryan(theta_x, theta_y, theta_z):
    C_x = rotation_matrix_x(theta_x)
    print_func(C_x, "C_x")
    C_y = rotation_matrix_y(theta_y)
    print_func(C_y, "C_y")
    C_z = rotation_matrix_z(theta_z)
    print_func(C_z, "C_z")

    C_iv_init = np.matmul(C_z, C_y) 
    C_iv = np.matmul(C_iv_init, C_x)
    return np.around(C_iv, 3)

def transformation_matrix(C, x_vi, y_vi, z_vi):
    T_iv = np.zeros((4,4))
    T_iv[0,0] = C[0,0]
    T_iv[0,1] = C[0,1]
    T_iv[0,2] = C[0,2]
    T_iv[1,0] = C[1,0]
    T_iv[1,1] = C[1,1]
    T_iv[1,2] = C[1,2]
    T_iv[2,0] = C[2,0]
    T_iv[2,1] = C[2,1]
    T_iv[2,2] = C[2,2]
    T_iv[0,3] = x_vi
    T_iv[1,3] = y_vi
    T_iv[2,3] = z_vi
    T_iv[3,3] = 1
    return np.around(T_iv, 3)

def print_func(x, text_x):
    print(text_x)
    print(x)

def transform_point(T, p):
    p_hom = np.ones((4, np.shape(p)[1]))
    p_hom[0:np.shape(p)[0], :] = p[0:np.shape(p)[0], :]
    # print("p_hom")
    # print(p_hom)
    p_transformed_hom =  np.matmul(T, p_hom)
    p_transformed = p_transformed_hom[0:3, :]
    return p_transformed

def normalized_image_plane_projection(pose):
    pose_norm = np.ones((3, np.shape(pose)[1]))
    for i in range(np.shape(pose)[1]):
        pose_norm[0, i] = pose[0, i]/pose[2, i]
        pose_norm[1, i] = pose[1, i]/pose[2, i]
    return pose_norm

def lens_distortion(pose_norm, k_1, k_2, k_3, T_1, T_2):
    pose_distort = np.ones((3, np.shape(pose_norm)[1]))
    for i in range(np.shape(pose_norm)[1]):
        x_n = pose_norm[0, i]
        y_n = pose_norm[1, i]
        r = math.sqrt(pow(x_n, 2) + pow(y_n, 2))
        pose_distort[0, i] = (1 + k_1*pow(r, 2) + k_2*pow(r, 4) + k_3*pow(r, 6)) * x_n + 2*T_1*x_n*y_n + T_2*(pow(r, 2) + 2*pow(x_n, 2))
        pose_distort[1, i] = (1 + k_1*pow(r, 2) + k_2*pow(r, 4) + k_3*pow(r, 6)) * y_n + 2*T_2*x_n*y_n + T_1*(pow(r, 2) + 2*pow(y_n, 2))
    return pose_distort

def pixel_coordinates(pose_distort, f_x, f_y, c_x, c_y):
    pixel_matrix = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    pixel_coord_hom = np.matmul(pixel_matrix, pose_distort)
    pixel_coord = np.delete(pixel_coord_hom, 2, 0)
    return np.around(pixel_coord, 0)

def calculate_depth(pose):
    depth_matrix = np.zeros((1, np.shape(pose)[1]))
    for i in range(np.shape(pose)[1]):
        depth_matrix[0, i] = math.sqrt(pow(pose[0, i], 2) + pow(pose[1, i], 2) + pow(pose[2, i], 2))
    return depth_matrix

def pixel_coordinates_based_on_resolution(pixel_coord, x_c_lim, y_c_lim):
    index_to_del = []
    for i in range(np.shape(pixel_coord)[1]):
        if (pixel_coord[0, i] > x_c_lim) or (pixel_coord[0, i] < 0) or (pixel_coord[1, i] > y_c_lim) or (pixel_coord[1, i] < 0):
            index_to_del.append(i)
    pixel_coord_updated = np.delete(pixel_coord, index_to_del, 1)
    
    return pixel_coord_updated