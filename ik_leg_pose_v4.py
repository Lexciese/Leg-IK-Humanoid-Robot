import math
import numpy as np
import PyKDL as kdl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial import distance

def set_label(ax, min_scale=-10, max_scale=10):
    ax.set_xlim(min_scale, max_scale)
    ax.set_ylim(min_scale, max_scale)
    ax.set_zlim(min_scale, max_scale)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

def draw_axis(ax, scale=1.0, O=np.eye(4), style='-'):
    xaxis = np.array([[0, 0, 0, 1], [scale, 0, 0, 1]]).T
    yaxis = np.array([[0, 0, 0, 1], [0, scale, 0, 1]]).T
    zaxis = np.array([[0, 0, 0, 1], [0, 0, scale, 1]]).T
    xc = O.dot(xaxis)
    yc = O.dot(yaxis)
    zc = O.dot(zaxis) 
    ax.plot(xc[0,:], xc[1,:], xc[2,:], 'r' + style)
    ax.plot(yc[0,:], yc[1,:], yc[2,:], 'g' + style)
    ax.plot(zc[0,:], zc[1,:], zc[2,:], 'b' + style)
    
def RX(yaw):
    return np.array([[1, 0, 0], 
                     [0, math.cos(yaw), -math.sin(yaw)], 
                     [0, math.sin(yaw), math.cos(yaw)]])   

def RY(delta):
    return np.array([[math.cos(delta), 0, math.sin(delta)], 
                     [0, 1, 0], 
                     [-math.sin(delta), 0, math.cos(delta)]])

def RZ(theta):
    return np.array([[math.cos(theta), -math.sin(theta), 0], 
                     [math.sin(theta), math.cos(theta), 0], 
                     [0, 0, 1]])

def TF(rot_axis=None, q=0, dx=0, dy=0, dz=0):
    if rot_axis == 'x':
        R = RX(q)
    elif rot_axis == 'y':
        R = RY(q)
    elif rot_axis == 'z':
        R = RZ(q)
    elif rot_axis == None:
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    
    T = np.array([[R[0,0], R[0,1], R[0,2], dx],
                  [R[1,0], R[1,1], R[1,2], dy],
                  [R[2,0], R[2,1], R[2,2], dz],
                  [0, 0, 0, 1]])
    return T

def draw_links(ax, origin_frame=np.eye(4), target_frame=np.eye(4)):
    x = [origin_frame[0,3], target_frame[0,3]]
    y = [origin_frame[1,3], target_frame[1,3]]
    z = [origin_frame[2,3], target_frame[2,3]]
    ax.plot(x, y, z, 'k')

def pose_error(f_target, f_result):
    f_diff = f_target.Inverse() * f_result
    [dx, dy, dz] = f_diff.p
    [drz, dry, drx] = f_diff.M.GetEulerZYX()
    error = np.sqrt(dx**2 + dy**2 + dz**2 + drx**2 + dry**2 + drz**2)
    error_list = [dx, dy, dz, drx, dry, drz]
    return error, error_list

def calculate_leg_kinematics(x, y, z, yaw, CROTCH_TO_HIP, UPPER_HIP, HIP_TO_KNEE, KNEE_TO_ANKLE, ANKLE_TO_SOLE, invert=False):
    x_from_hip = (x - 0)
    y_from_hip = (y - CROTCH_TO_HIP)
    z_from_hip = (z + (UPPER_HIP + ANKLE_TO_SOLE))

    xa = x_from_hip
    ya = xa * np.tan(yaw)
    beta = np.pi/2 - yaw
    yb = (y_from_hip - ya)
    gamma = np.pi/2 - beta
    xb = xa / np.cos(yaw) + np.sin(gamma) * (y_from_hip - ya)
    x_from_hip_yaw = xb
    y_from_hip_yaw = yb
    z_from_hip_yaw = z_from_hip

    C = np.sqrt(xb**2 + yb**2 + z_from_hip_yaw**2)
    zb = np.sqrt(yb**2 + z_from_hip_yaw**2)
    zc = np.sqrt(xb**2 + z_from_hip_yaw**2)
    zeta = np.arctan2(yb, zc)
    Cb = np.sign(xb)*np.sqrt(C**2 - zb**2)

    q_hip_yaw = yaw
    q_hip_roll = zeta 
    q_hip_pitch = -(np.arctan2(Cb, np.sign(z_from_hip_yaw)*z_from_hip_yaw) + np.arccos((C/2)/HIP_TO_KNEE))
    q_knee = np.pi-(2*(np.arcsin((C/2)/HIP_TO_KNEE)))
    q_ankle_pitch = -(np.pi/2 - (np.arctan2(np.sign(z_from_hip_yaw)*z_from_hip_yaw, Cb) + np.arccos((C/2)/HIP_TO_KNEE)))
    q_ankle_roll = -q_hip_roll

    if invert:
        q_hip_yaw = -q_hip_yaw
        q_hip_roll = -q_hip_roll
        q_hip_pitch = -q_hip_pitch
        q_ankle_pitch = -q_ankle_pitch
        q_ankle_roll = -q_ankle_roll
    
    print("hip_yaw :", (q_hip_yaw))
    print("q_hip_roll", q_hip_roll)
    print("hip_roll :", (q_hip_roll))
    print("cb ", Cb)
    print("hip_pitch :", (q_hip_pitch))
    print("knee :", (q_knee))
    print("ankle_pitch :", (q_ankle_pitch))
    print("ankle_roll :", (q_ankle_roll))
    
    print("=====================================")

    return q_hip_yaw, q_hip_roll, q_hip_pitch, q_knee, q_ankle_pitch, q_ankle_roll

def plot_leg(ax, q_hip_yaw, q_hip_roll, q_hip_pitch, q_knee, q_ankle_pitch, q_ankle_roll, CROTCH_TO_HIP, UPPER_HIP, HIP_TO_KNEE, KNEE_TO_ANKLE, ANKLE_TO_SOLE, base_offset_y=0):
    # Base frame with offset
    base = TF(dy=base_offset_y)

    # Hip yaw transformation
    hip_yaw_from_base = TF(rot_axis='z', q=q_hip_yaw, dy=CROTCH_TO_HIP)
    hip_yaw = base.dot(hip_yaw_from_base)

    # Hip roll transformation
    hip_roll_from_hip_yaw = TF(rot_axis='x', q=q_hip_roll, dz=-UPPER_HIP)
    hip_roll = hip_yaw.dot(hip_roll_from_hip_yaw)

    # Hip pitch transformation
    hip_pitch_from_hip_roll = TF(rot_axis='y', q=q_hip_pitch)
    hip_pitch = hip_roll.dot(hip_pitch_from_hip_roll)
    hip = hip_pitch  # Hip frame

    # Knee transformation
    knee_from_hip = TF(rot_axis='y', q=q_knee, dz=-HIP_TO_KNEE)
    knee = hip.dot(knee_from_hip)  # Knee frame

    # Ankle pitch transformation
    ankle_pitch_from_knee = TF(rot_axis='y', q=q_ankle_pitch, dz=-KNEE_TO_ANKLE)
    ankle_pitch = knee.dot(ankle_pitch_from_knee)

    # Ankle roll transformation
    ankle_roll_from_ankle_pitch = TF(rot_axis='x', q=q_ankle_roll)
    ankle = ankle_pitch.dot(ankle_roll_from_ankle_pitch)  # Ankle frame

    # Sole transformation
    sole_from_ankle = TF(dz=-ANKLE_TO_SOLE)
    sole = ankle.dot(sole_from_ankle)  # Sole frame

    # Draw the leg
    draw_axis(ax, scale=0.050, O=base)
    draw_links(ax, origin_frame=base, target_frame=hip_yaw)
    draw_axis(ax, scale=0.050, O=hip_yaw)
    draw_links(ax, origin_frame=hip_yaw, target_frame=hip)
    draw_axis(ax, scale=0.050, O=hip)
    draw_links(ax, origin_frame=hip, target_frame=knee)
    draw_axis(ax, scale=0.050, O=knee)
    draw_links(ax, origin_frame=knee, target_frame=ankle)
    draw_axis(ax, scale=0.050, O=ankle)
    draw_links(ax, origin_frame=ankle, target_frame=sole)
    draw_axis(ax, scale=0.050, O=sole)

def main():
    epsilon = np.finfo(np.float32).eps
    err_min = 1e-2
    # Konfigurasi link in mm
    CROTCH_TO_HIP = 0.035 # Jarak croth ke hip
    UPPER_HIP = 0.050 # Jarak hip yaw ke hip roll pitch
    HIP_TO_KNEE = 0.2215 # Panjang link upper leg
    KNEE_TO_ANKLE = 0.2215 # Panjang link lower leg
    ANKLE_TO_SOLE = 0.053 # Jarak ankle ke sole

    # Input posisi for right leg
    x_r = 0.000
    y_r = -0.035
    z_r = -0.273
    yaw_r = np.radians(0)

    # Input posisi for left leg
    x_l = 0.000
    y_l = 0.035
    z_l = -0.273
    yaw_l = np.radians(0)

    # Calculate kinematics for right leg
    q_hip_yaw_r, q_hip_roll_r, q_hip_pitch_r, q_knee_r, q_ankle_pitch_r, q_ankle_roll_r = calculate_leg_kinematics(
        x_r, y_r, z_r, yaw_r, CROTCH_TO_HIP, UPPER_HIP, HIP_TO_KNEE, KNEE_TO_ANKLE, ANKLE_TO_SOLE, invert=True)

    # Calculate kinematics for left leg with inverted angles
    q_hip_yaw_l, q_hip_roll_l, q_hip_pitch_l, q_knee_l, q_ankle_pitch_l, q_ankle_roll_l = calculate_leg_kinematics(
        x_l, y_l, z_l, yaw_l, CROTCH_TO_HIP, UPPER_HIP, HIP_TO_KNEE, KNEE_TO_ANKLE, ANKLE_TO_SOLE, invert=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot right leg
    # each joint angle is "inverted again* from the result to match the axis direction. Alternative to plot the right leg properly
    plot_leg(ax, -q_hip_yaw_r, -q_hip_roll_r, -q_hip_pitch_r, q_knee_r, -q_ankle_pitch_r, -q_ankle_roll_r, 
             CROTCH_TO_HIP, UPPER_HIP, HIP_TO_KNEE, KNEE_TO_ANKLE, ANKLE_TO_SOLE, base_offset_y=-CROTCH_TO_HIP)

    # Plot left leg
    plot_leg(ax, q_hip_yaw_l, q_hip_roll_l, q_hip_pitch_l, q_knee_l, q_ankle_pitch_l, q_ankle_roll_l, 
             CROTCH_TO_HIP, UPPER_HIP, HIP_TO_KNEE, KNEE_TO_ANKLE, ANKLE_TO_SOLE, base_offset_y=CROTCH_TO_HIP)

    ax.auto_scale_xyz([-0.500, 0.500], [-0.500, 0.500], [-0.500, 0.500])
    plt.show()
    
if __name__ == "__main__":
    main()