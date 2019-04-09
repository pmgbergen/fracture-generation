#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:31:43 2018

@author: eke001
"""
import scipy.stats as stats
import numpy as np


class Signum:
    def __init__(self, fraction_negative=0.5):
        self.fraction_negative = fraction_negative

    def rvs(self, size, **kwagrs):
        vals = np.random.rand(size)
        tmp = np.ones(size)
        tmp[vals < self.fraction_negative] = -1
        return tmp

class Uniform:
    def __init__(self, value=0):
        self.value = value

    def rvs(self, size, **kwagrs):
        return self.value * np.ones(size)
