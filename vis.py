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


def plot_actuation(iterations, actuation_data, path):
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5)) 
    x = iterations
    legends = ['P_B', 'P_R', 'P_theta', 'x', 'y']
    
    for i in range(actuation_data.shape[0]):
        y = actuation_data[i]
        plt.plot(x, y) 
    
    ax.grid()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Actuation')
    ax.set_title('Actuation vs Iteration')  
    plt.tight_layout()
    plt.legend(legends)
    
    if not os.path.exists(os.path.join('images/', path)):
        os.makedirs(os.path.join('images/', path))
    plt.savefig(os.path.join('images/', path, 'actuation_curve.png'))
    

def plot_superimposition(img1_path, img2_path, path):

    background = Image.open(os.path.join('images/', path, img1_path))
    overlay = Image.open(os.path.join('images/', path, img2_path))
    
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    
    new_img = Image.blend(background, overlay, 0.5)
    new_img.save(os.path.join('images/', path, 'superimposition.png'),"PNG")
    
    # probably plot current image, final image and superimposition in subplots
    

def plot_mse_error(current_images_path, final_image_path, path):
    
    iterations = [x for x in range(len(current_images_path))]

    mse_list = []
    img_final = Image.open(os.path.join('images/', path, final_image_path))
    for x in current_images_path:
        
        img1 = Image.open(os.path.join('images/', path, x))
        
        mse = compare_mse(img_as_float(img1), img_as_float(img_final))
        
        mse_list.append(mse)
        
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5)) 
    
    plt.plot(iterations, mse_list)
    
    ax.grid()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')
    ax.set_title('MSE vs Iteration')  
    plt.tight_layout()
    
    plt.savefig(os.path.join('images/', path, 'msecurve.png'))
        
    
    

num_iterations = 10

iterations = [x for x in range(num_iterations)]

actuation_data = np.random.rand(5, num_iterations)

path = 'exp1/'

img1_path = 'IMG_1291.jpg'
img2_path = 'IMG_1292.jpg'

plot_actuation(iterations, actuation_data, path)

plot_superimposition(img1_path, img2_path, path)