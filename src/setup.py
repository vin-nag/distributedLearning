#!/usr/bin/env python

from distutils.core import setup

setup(name='Distributed DNN Learning',
      version='1.0',
      description='Train DNNs using RPCs (Apache Thrift) and PyTorch',
      author='Vineel Nagisetty',
      author_email='vineel.nagisetty@uwaterloo.ca',
      packages=['.'],
     )