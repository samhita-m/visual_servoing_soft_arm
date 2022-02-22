
import math
import os
import pandas

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager


from PIL import Image


plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

experiments = ['/Integrated/', '/Modular/', '/New Targets/', '/Light Intensity/', '/Diminution/', '/Uniform Load/']


gt_indices = [11, 13, 7, 4, 7, 9]

current_folder = os.getcwd()

test_sub_folder = current_folder + '/final_results_data/'


save_plots_path = current_folder + '/rebuttal_plots' 
    


fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)


plot_colors = ['#CC6677', '#332288', '#117733', '#882255', '#44AA99', '#999933', '#EE3377']  ## '#88CCEE', 

#plot_colors = ['#44AA99', '#999933']  ## '#88CCEE', 

lw = 5
image_format = 'svg'


def delq(q1, q2):
    
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    dq = w1*w2 + x1*x2 + y1*y2 + z1*z2
    dx = w1*x2 - x1*w2 + y1*z2 - z1*y2
    dy = w1*y2 - y1*w2 - x1*z2 + z1*x2
    dz = w1*z2 - w2*z1 + x1*y2 - y1*x2
    
    t1 = np.linalg.norm(np.array([dx, dy, dz]))
    
    theta_rad = 2*np.arctan2(t1, dq)
    theta_deg = np.rad2deg(theta_rad)
    
    # print(np.linalg.norm(np.array([dq, dx, dy, dz])))
    
  #  print( theta_rad, theta_deg)

    
    return theta_rad, theta_deg


def quat2euler(qw, qx, qy, qz):
    
     
    # First row of the rotation matrix
    r00 = 1 - 2 * ((qy * qy) + (qz * qz))
    r01 = 2 * ((qx * qy) - (qz * qw))
    r02 = 2 * ((qx * qz) + (qy * qw))
     
    # Second row of the rotation matrix
    r10 = 2 * ((qx * qy) + (qz * qw))
    r11 = 1 - 2 * ((qx * qx) + (qz * qz))
    r12 = 2 * ((qy * qz) - (qx * qw))
     
    # Third row of the rotation matrix
    r20 = 2 * ((qx * qz) - (qy * qw))
    r21 = 2 * ((qy * qz) + (qx * qw))
    r22 = 1 - 2 * ((qx * qx) + (qy * qy))
     
    R = np.array([[r00, r01, r02],
                  [r10, r11, r12],
                  [r20, r21, r22]])
    
                            
    return R





N = len(experiments)
for i in range(len(experiments)):


   
    test_sub_folders  = test_sub_folder +  experiments[i] 
    
    ground_truth_data = pandas.read_csv(test_sub_folders + '/points_data_1.csv')
    
    pos_array = []
    rot_array = []
    rot_array2 = []
    
    gt_index = gt_indices[i]
    
    
    csv_data = pandas.read_csv(test_sub_folders + '/data_1.csv')    
    iterations = csv_data['It No']
    
    for j in range(len(iterations)):
        
        
        
        px = abs(csv_data['Px'][j] - ground_truth_data['Px'][gt_index])
        py = abs(csv_data['Py'][j] - ground_truth_data['Py'][gt_index])
        pz = abs(csv_data['Pz'][j] - ground_truth_data['Pz'][gt_index])
        
        p_error = math.sqrt(px**2 + py**2 + pz**2) ## L2 norm
        
        q0_fin = csv_data['q0'][j]
        q1_fin = csv_data['q1'][j]
        q2_fin = csv_data['q2'][j]
        q3_fin = csv_data['q3'][j]
        
        fin_norm = np.linalg.norm(np.array([q0_fin, q1_fin, q2_fin, q3_fin]))
            
        q0_fin = q0_fin/fin_norm
        q1_fin = q1_fin/fin_norm
        q2_fin = q2_fin/fin_norm
        q3_fin = q3_fin/fin_norm
        
        
        q0_gt = ground_truth_data['q0'][gt_index]
        q1_gt = ground_truth_data['q1'][gt_index]
        q2_gt = ground_truth_data['q2'][gt_index]
        q3_gt = ground_truth_data['q3'][gt_index]
        
        gt_norm = np.linalg.norm(np.array([q0_gt, q1_gt, q2_gt, q3_gt]))
        
        q0_gt = q0_gt/gt_norm
        q1_gt = q1_gt/gt_norm
        q2_gt = q2_gt/gt_norm
        q3_gt = q3_gt/gt_norm
        
        
        Q1 = [q0_gt, q1_gt, q2_gt, q3_gt]
        Q2 = [q0_fin, q1_fin, q2_fin, q3_fin]
        
        r_error_rad, r_error_deg = delq(Q1, Q2)
        
        R_fin = quat2euler(q0_fin, q1_fin, q2_fin, q3_fin)
        R_gt = quat2euler(q0_gt, q1_gt, q2_gt, q3_gt)
        
        
        trace = np.trace(np.matmul(R_fin, np.transpose(R_gt)))    
        trace = np.clip((trace - 1)/2.0, -1, 1)
        
        r_error = np.arccos(trace)
        
        
        
        
        pos_array.append(p_error)
        rot_array.append(r_error)
        rot_array2.append(r_error_rad)
    
   
   
    
    ax2.plot(iterations, pos_array, linewidth=lw, color = plot_colors[i] )
    ax2.set_ylim([-0.01, 12])
    ax2.grid(b=True)

    ax3.plot(iterations, rot_array, linewidth=lw, color = plot_colors[i] )
    ax3.set_ylim([-0.01, 1.5])
    ax3.grid(b=True)
    
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
	    label.set_fontsize(20)
        
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
	    label.set_fontsize(20)
   
    

    
    
   # plt.savefig(save_plots_path + '/plots' + str(i) + '_' + '.jpg', bbox_inches='tight', dpi=200)
    
   



ax2.set_xlabel('Iterations', fontsize=25)
ax3.set_xlabel('Iterations', fontsize=25)


ax2.set_ylabel('Pos Error (cm)', fontsize=25)
ax3.set_ylabel('Rot Error (rad)', fontsize=25)


legend_list = [x[1:-1] for x in experiments]


fig2.legend(legend_list,  loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=1, prop={'size':25})


fig2.savefig(save_plots_path + '/pos_error.svg', bbox_inches='tight', format=image_format, dpi=1200)
        

fig3.savefig(save_plots_path + '/rot_error.svg', bbox_inches='tight', format=image_format, dpi=1200)



fig2.savefig(save_plots_path + '/pos_error.png', bbox_inches='tight',  dpi=1200)
        
fig3.savefig(save_plots_path + '/rot_error.png', bbox_inches='tight',  dpi=1200)

