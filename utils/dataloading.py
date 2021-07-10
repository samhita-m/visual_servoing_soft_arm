#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:37:42 2021

@author: samhita
"""




# Sorting 

images = [] # load the image list here


# Adding zeros ---> Do this for all files in the list

file_name = 'img1.png'
file_num = file_name[3:]
file_num = file_num[:-4]

j = file_num.zfill(4) # zfill input = 4 converts 1 to 0001 ---> change this as needed
file_name_modified = 'img' + j + '.png'


images.sort() 
sorted(images)  


