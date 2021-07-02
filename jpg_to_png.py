#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 00:33:03 2021

@author: samhita
"""


import os
from PIL import Image
from skimage import img_as_float

img1_path = '/home/samhita/visual_servoing_soft_arm/images/test2/test214.jpg'

background = Image.open( img1_path)



background.save('/home/samhita/visual_servoing_soft_arm/images/test2/test214.png',"PNG")