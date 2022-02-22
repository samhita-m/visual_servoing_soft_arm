
import math
import os
import pandas

import numpy as np
import matplotlib.pyplot as plt


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



#quaternion to euler angle conversion
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


plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']




current_folder = os.getcwd()


mse_convergence = 0.05
attempt_folder = 'Attempt_0_05' # -3.75 -3
save_plots_path = current_folder + '/rebuttal_plots'

experiments = ['/Integrated/', '/Modular/', '/New Targets/', '/Light Intensity/', '/Diminution/', '/Uniform Load/']

sub = ['TIM', 'PD_', 'NT_', 'IM', 'bind_', 'weight_']

attempt_folder = ['Attempt2', '', 'Attempt_0_05', 'Attempt_0_05', 'Attempt_0_05' , 'Attempt_0_05']

range_start = [0, 0, 1, 0, 0, 0]


i_list = [range(30),
          range(15), 
          [0, 1, 3, 4, 5, 7],
          range(10), 
          range(10), 
          range(10)]

plot_colors = ['#CC6677', '#332288', '#117733', '#882255', '#44AA99', '#999933', '#EE3377']  ## '#88CCEE', 



pos_arrays = []
rot_arrays = []
rot_arrays2 = []



    
for i in range(len(experiments)):


    ground_truth = current_folder + experiments[i] + 'completed'
    test_sub_folders  = current_folder +  experiments[i] + sub[i]
    
    ground_truth_data = pandas.read_csv(ground_truth + '/points_data_1.csv')
    
    pos_array = []
    rot_array = []
    rot_array2 = []
    
    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5))

    fig4, ax4 = plt.subplots(1, 1, figsize=(6, 5))

    fig5, ax5 = plt.subplots(1, 1, figsize=(6, 5))
    
    
    for j in i_list[i]:
        
        sub_folder = test_sub_folders + str(j) + '/' + attempt_folder[i]  
        csv_data = pandas.read_csv(sub_folder + '/data_1.csv')    
        iterations = csv_data['It No']
        
       
            
        
        px = abs(csv_data['Px'][len(iterations)-1] - ground_truth_data['Px'][j])
        py = abs(csv_data['Py'][len(iterations)-1] - ground_truth_data['Py'][j])
        pz = abs(csv_data['Pz'][len(iterations)-1] - ground_truth_data['Pz'][j])
        
        p_error = math.sqrt(px**2 + py**2 + pz**2) ## L2 norm
        
        q0_fin = csv_data['q0'][len(iterations)-1]
        q1_fin = csv_data['q1'][len(iterations)-1]
        q2_fin = csv_data['q2'][len(iterations)-1]
        q3_fin = csv_data['q3'][len(iterations)-1]
        
        fin_norm = np.linalg.norm(np.array([q0_fin, q1_fin, q2_fin, q3_fin]))
            
        q0_fin = q0_fin/fin_norm
        q1_fin = q1_fin/fin_norm
        q2_fin = q2_fin/fin_norm
        q3_fin = q3_fin/fin_norm
        
        
        q0_gt = ground_truth_data['q0'][j]
        q1_gt = ground_truth_data['q1'][j]
        q2_gt = ground_truth_data['q2'][j]
        q3_gt = ground_truth_data['q3'][j]
        
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
    #    rot_array2.append(r_error_deg)
      
        
    pos_arrays.append(pos_array)
    rot_arrays.append(rot_array)
    rot_arrays2.append(rot_array2)
    

    

    p_mean = np.mean(pos_array)
    p_std = np.std(pos_array)
    q_mean = np.mean(rot_array)
    q_std = np.std(rot_array) 
    
  
    
    props = dict(boxstyle='round', facecolor='grey', alpha=0.2)
    
    
    
    bins=[0, 1, 2, 3, 4]
    thresh = 3
    bin0 = 0
    bin1 = 0
    bin2 = 0
    bin3 = 0
    for k in range(len(pos_array)):    
        if pos_array[k] >=0 and pos_array[k] < 1:
            bin0 += 1        
        elif pos_array[k] >=1 and pos_array[k] < 2:
            bin1 += 1        
        elif pos_array[k] >=2 and pos_array[k] < 3:        
            bin2 +=1        
        else:        
            bin3 +=1
            
    bin_array = [bin0, bin1, bin2, bin3]
    
    bin_percent = [x/sum(bin_array)*100 for x in bin_array]    
    
    
    ax3.bar(bins[:-1], bin_percent, width=1, align="edge",  facecolor = '#2ab0ff', edgecolor='k', label=experiments[i][1:-1])
    
    ax3.yaxis.set_ticks(np.arange(0, 110, 20))
    vals = ax3.get_yticks()
    ax3.set_yticklabels(['{:,.0%}'.format(x/100) for x in vals])
    ax3.xaxis.set_ticks(np.arange(0, 4, 1))
    ax3.tick_params(axis='both',  labelsize=16)
    
    ax3.text(3.5, -7, '>3', fontsize=16, color='#CC3311')
    ax3.set_xlabel('Translation Error (cm)', fontsize=16)
    
    
    image_format = 'svg' 
    
    
    
    
        
    bins=[0, 0.06, 0.12, 0.18, 0.24]

    thresh = 1

    bin0 = 0
    bin1 = 0
    bin2 = 0
    bin3 = 0
    bin4 = 0
    for k in range(len(rot_array)):    
        if rot_array[k] >= bins[0] and rot_array[k] <= bins[1]:
            bin0 += 1        
        elif rot_array[k] > bins[1] and rot_array[k] <= bins[2]:
            bin1 += 1        
        elif rot_array[k] > bins[2] and rot_array[k] <= bins[3]:        
            bin2 +=1    
        elif rot_array[k] > bins[3] and rot_array[k] <= bins[4]:        
            bin3 +=1  
        
        elif rot_array[k] > bins[4]:        
            bin4 +=1
            
    bin_array = [bin0, bin1, bin2, bin3, bin4]
    
    bin_percent = [x/sum(bin_array)*100 for x in bin_array]  
    
    
    ax4.bar(bins,  bin_percent, width=0.06, align="edge",  facecolor = '#BB5566', edgecolor='k', label=experiments[i][1:-1])
    
    ax4.yaxis.set_ticks(np.arange(0, 110, 20))
    vals = ax4.get_yticks()
    ax4.set_yticklabels(['{:,.0%}'.format(x/100) for x in vals])
    ax4.xaxis.set_ticks(np.arange(0, 0.30, 0.06))
    ax4.tick_params(axis='both',  labelsize=16)
    
    ax4.text(0.26, -7, '>0.24', fontsize=16, color='#CC3311')
    ax4.set_xlabel('Rotation Error (rad)', fontsize=16)
    
   # print(rot_array)
        
  #  print(bin_array)
  
    
  
    bins=[0, 0.06, 0.12, 0.18, 0.24]

    thresh = 1

    bin0 = 0
    bin1 = 0
    bin2 = 0
    bin3 = 0
    bin4 = 0
    for k in range(len(rot_array)):    
        if rot_array2[k] >= bins[0] and rot_array2[k] <= bins[1]:
            bin0 += 1        
        elif rot_array2[k] > bins[1] and rot_array2[k] <= bins[2]:
            bin1 += 1        
        elif rot_array2[k] > bins[2] and rot_array2[k] <= bins[3]:        
            bin2 +=1    
        elif rot_array2[k] > bins[3] and rot_array2[k] <= bins[4]:        
            bin3 +=1  
        
        elif rot_array2[k] > bins[4]:        
            bin4 +=1
            
    bin_array = [bin0, bin1, bin2, bin3, bin4]
    
    bin_percent = [x/sum(bin_array)*100 for x in bin_array]  
    
    
    ax5.bar(bins,  bin_percent, width=0.06, align="edge",  facecolor = '#BB5566', edgecolor='k', label=experiments[i][1:-1])
    
    
    
    ax5.yaxis.set_ticks(np.arange(0, 110, 20))
    vals = ax4.get_yticks()
    ax5.set_yticklabels(['{:,.0%}'.format(x/100) for x in vals])
    ax5.xaxis.set_ticks(np.arange(0, 0.30, 0.06))
    ax5.tick_params(axis='both',  labelsize=16)
    
    ax5.text(0.26, -7, '>0.24', fontsize=16, color='#CC3311')
    ax5.set_xlabel('Rotation Error (deg)', fontsize=16)
      
  
  
    '''
    
    bins = [0, 2, 4, 6, 8, 10]
    thresh = 1

    bin0 = 0
    bin1 = 0
    bin2 = 0
    bin3 = 0
    bin4 = 0
  
    bin5 = 0
    for k in range(len(rot_array2)):    
        if rot_array2[k] >= bins[0] and rot_array2[k] <= bins[1]:
            bin0 += 1        
        elif rot_array2[k] > bins[1] and rot_array2[k] <= bins[2]:
            bin1 += 1        
        elif rot_array2[k] > bins[2] and rot_array2[k] <= bins[3]:        
            bin2 +=1    
        elif rot_array2[k] > bins[3] and rot_array2[k] <= bins[4]:        
            bin3 +=1  
        elif rot_array2[k] > bins[4] and rot_array2[k] <= bins[5]:        
            bin4 +=1    
       
        elif rot_array2[k] > bins[5]:        
            bin5 +=1
            
    bin_array = [bin0, bin1, bin2, bin3, bin4, bin5]
    
   # print(rot_array2)
    
   # print(bin_array)
    
    bin_percent = [x/sum(bin_array)*100 for x in bin_array]    
    
    
    
   
    
    ax5.bar(bins, bin_percent, width=2, align="edge", facecolor = '#BB5566', edgecolor='k', label=experiments[i][1:-1])
    
    ax5.yaxis.set_ticks(np.arange(0, 110, 10))
    vals = ax4.get_yticks()
    ax5.set_yticklabels(['{:,.0%}'.format(x/100) for x in vals])
    ax5.xaxis.set_ticks(np.arange(0, 12, 2))
    ax5.tick_params(axis='both',  labelsize=12)
    
    ax5.text(10.75, -6.25 , '>10', fontsize=12, color='#CC3311')
    ax5.set_xlabel('Rotation Error (deg)', fontsize=12)
    
    '''
    
   
        
    fig3.savefig(save_plots_path + '/histogram_pos' + experiments[i][1:-1] + '.svg', bbox_inches='tight', format=image_format, dpi=1200)    
        
    
    fig4.savefig(save_plots_path + '/histogram_rot_bind_rad' + experiments[i][1:-1] + '.svg', bbox_inches='tight', format=image_format, dpi=1200)
    
    fig5.savefig(save_plots_path + '/histogram_rot_bind_deg' + experiments[i][1:-1] + '.svg', bbox_inches='tight', format=image_format, dpi=1200)




'''

#plots the histogram data
width = (bins[1] - bins[0]) * 0.4
bins_shifted = bins + width
ax1.bar(bins[:-1], n[0], width, align='edge', color=plot_colors[i])
ax1.bar(bins_shifted[:-1], n[1], width, align='edge', color=colors[1])

#finishes the plot
ax1.set_ylabel("Count1", color=colors[0])

ax1.set_xlabel('Rotation Error (deg)', fontsize=16)

ax1.tick_params('y', colors=colors[0])

plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

#fig4.savefig(save_plots_path + '/histogram_rot_bind' + '_' + attempt_folder + '.svg', bbox_inches='tight', format=image_format, dpi=1200)




'''


