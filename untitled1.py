# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:15:50 2024

@author: OM
"""

## Detect mirror symmetry

#### Author: yiran jing
#### Date: Feb 2020 


import sys
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
from untitled2 import * # class and helper function


def main():
    detecting_mirrorLine('occlusion2.png', "butterflywith Mirror Line", show_detail = True)
    
if __name__ == '__main__':
    main()