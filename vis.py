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

from skimage.measure import compare_mse

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
    
    
    

def plot_superimposition(img1_path, img2_path, results_path):

    background = Image.open( img1_path)
    overlay = Image.open( img2_path)
    
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    
    new_img = Image.blend(background, overlay, 0.5)
    new_img.save(os.path.join(results_path, 'superimposition.png'),"PNG")
    
    # probably plot current image, final image and superimposition in subplots
    

def plot_mse_error(current_images_path, final_image_path, results_path):
    
    iterations = [x for x in range(len(current_images_path))]

    mse_list = []
    img_final = Image.open(final_image_path)
    for i in range(len(current_images_path)):
        
        
        img1 = Image.open(current_images_path[len(current_images_path) - 1 - i])
        
        print(current_images_path[len(current_images_path) - 1 - i])
        
        mse = compare_mse(img_as_float(img1), img_as_float(img_final))
        
        mse_list.append(mse)
        
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5)) 
    
    plt.plot(iterations, mse_list)
    
    ax.grid()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')
    ax.set_title('MSE vs Iteration')  
    plt.tight_layout()
    
    plt.savefig(os.path.join(results_path, 'msecurve.png'))
        
    






path = 'test1/'

results_path = os.path.join('images/', path, 'results/')

if not os.path.exists(results_path):
    os.makedirs(results_path)
    

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


images_path = os.path.join('images/', path)
images = [os.path.join(images_path, file) for file in os.listdir(images_path) if file.endswith('.png')]


img_ini = images[-2]
img_fin = images[0]
img_tar = images[-1]


num_iterations = actuation_data.shape[0]

iterations = [x for x in range(num_iterations)]
 
 
plot_actuation(iterations, actuation_data, results_path)

plot_superimposition(img_fin, img_tar, results_path)


plot_mse_error(images[:-1], img_tar, results_path)


