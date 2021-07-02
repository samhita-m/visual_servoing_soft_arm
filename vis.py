#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:17:31 2021

@author: samhita
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import img_as_float

from skimage.metrics import mean_squared_error

def plot_actuation(iterations, actuation_data, results_path):
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5)) 
    x = iterations
    legends = ['P_B', 'P_R', 'P_theta', 'x', 'y']
    legends_use = []
    
    for i in range(actuation_data.shape[1]):
        y = actuation_data[:, i]
        plt.plot(x, y) 
        legends_use.append(legends[i])
    
    ax.grid()
    ax.set_xticks(iterations, minor=True)
    ax.grid(which="minor",alpha=0.3)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Actuation')
    ax.set_title('Actuation vs Iteration')  
    plt.tight_layout()
    plt.legend(legends)
    
    plt.savefig(os.path.join(results_path, 'actuation_curve.png'))
    
    
    

def plot_superimposition(final_image_path, target_image_path, results_path):

    background = Image.open(final_image_path)
    overlay = Image.open( target_image_path)
    
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    
    new_img = Image.blend(background, overlay, 0.5)
    new_img.save(os.path.join(results_path, 'superimposition.png'),"PNG")
    
    # probably plot current image, final image and superimposition in subplots
    
    
    
    

def plot_mse_error(current_images_path, target_image_path, results_path):
    
    iterations = [x for x in range(len(current_images_path))]

    mse_list = []
    img_final = Image.open(target_image_path)
    for x in current_images_path:
        
        img1 = Image.open(x)
                
        mse = mean_squared_error(img_as_float(img1), img_as_float(img_final))
        
        mse_list.append(mse)
        
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5)) 
    
    plt.plot(iterations, mse_list)
    
    ax.grid()
    ax.set_xticks(iterations)
    
    ax.grid(which="major", alpha=0.6)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')
    ax.set_title('MSE vs Iteration')  
    plt.tight_layout()
    
    plt.savefig(os.path.join(results_path, 'msecurve.png'))
        
    


path = 'test1/'
actuation_data = np.array([[27, -1, 1.],
                           [21, 3, 3], 
                           [22, -5, 0],
                           [20, -3, 1]])

         


'''
path = 'test2/'
actuation_data = np.array([[21, -3, 1], 
                           [25, -6, 4.], 
                           [21, -3, -1.], 
                           [22, -3, 0.],
                           [20, -5, 0.],
                           [21, -4, 0.],
                           [18, -3, 1.],
                           [18, -5, 0.],
                           [20,-3,0.],
                           [22, -4, 1.],
                           [20, -5, 0]])

'''    
           
'''
path = 'test3/'
actuation_data = np.array([[18, -20, 4], 
                           [24, -10, -1.], 
                           [22, -5, 0.],
                           [21, -4, 1.],
                           [20, -3, 0.],
                           [19, 14, 1.]])

'''

results_path = os.path.join('images/', path, 'results/')

if not os.path.exists(results_path):
    os.makedirs(results_path)
    

images_path = os.path.join('images/', path)
images = [os.path.join(images_path, file) for file in os.listdir(images_path) if file.endswith('.png')]

images.sort() 
sorted(images)

img_ini = images[1]
img_fin = images[-1]
img_tar = images[0]

num_iterations = actuation_data.shape[0]
iterations = [x for x in range(num_iterations)]
 
plot_actuation(iterations, actuation_data, results_path)

plot_superimposition(img_fin, img_tar, results_path)

plot_mse_error(images[1:], img_tar, results_path)



# Place all images in place

fig, ax = plt.subplots(1, 4, figsize=(10, 40))
img1 = Image.open(img_tar)
img2 = Image.open(img_ini)
img3 = Image.open(img_fin)
img4 = Image.open(os.path.join('images/', path, 'results/superimposition.png'))


ax[0].imshow(img1)
ax[0].tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)

ax[0].tick_params(
    axis='y',         
    which='both',    
    left=False,     
    right=False,        
    labelleft=False)

ax[0].set_xlabel('Target Image')

ax[1].imshow(img2)
ax[1].tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)

ax[1].tick_params(
    axis='y',         
    which='both',    
    left=False,     
    right=False,        
    labelleft=False)

ax[1].set_xlabel('Initial Image')

ax[2].imshow(img3)
ax[2].tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)

ax[2].tick_params(
    axis='y',         
    which='both',    
    left=False,     
    right=False,        
    labelleft=False)

ax[2].set_xlabel('Final Image')

ax[3].imshow(img4)
ax[3].tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)

ax[3].tick_params(
    axis='y',         
    which='both',    
    left=False,     
    right=False,        
    labelleft=False)

ax[3].set_xlabel('Superimposition Image')

plt.tight_layout()
plt.savefig(os.path.join(results_path, 'images_only.png'), bbox_inches='tight')

fig1, ax1 = plt.subplots(1, 2, figsize=(10, 20)) 

img_act = Image.open(os.path.join('images/', path, 'results/actuation_curve.png'))

img_mse = Image.open(os.path.join('images/', path, 'results/msecurve.png'))

ax1[0].imshow(img_act)
ax1[0].axis('off')
ax1[1].imshow(img_mse)
ax1[1].axis('off')


plt.tight_layout()
plt.savefig(os.path.join(results_path, 'iteration_trend.png'), bbox_inches='tight')


fig2, ax2 = plt.subplots(2, 1, figsize=(20, 15)) 

img_row1 = Image.open(os.path.join('images/', path, 'results/images_only.png'))

img_row2 = Image.open(os.path.join('images/', path, 'results/iteration_trend.png'))

ax2[0].imshow(img_row1)
ax2[0].axis('off')
ax2[1].imshow(img_row2)
ax2[1].axis('off')


plt.tight_layout()
plt.savefig(os.path.join(results_path, 'all_plots.png'), bbox_inches='tight')





