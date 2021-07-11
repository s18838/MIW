#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:06:02 2021

@author: taraskulyavets
"""

import numpy as np

v = np.array([1, 2, 3], dtype='int64')

print(v)


n = np.array([[1, 2, 3], [3.3, 4.3, 2.4]])

print(n)
print(n.ndim)
print(n.shape)
print(n.itemsize)

print(n.dtype)

