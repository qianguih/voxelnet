#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : setup.py
# Purpose :
# Creation Date : 11-12-2017
# Last Modified : Sat 23 Dec 2017 03:18:37 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]


from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='box overlaps',
    ext_modules=cythonize('./utils/box_overlaps.pyx')
)

# solution for potential error related to numpy/arrayobject.h
# export CFLAGS="-I /home/rcf-40/qianguih/.local/lib/python2.7/site-packages/numpy/core/include $CFLAGS"
