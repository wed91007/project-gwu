# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 15:14:56 2019

@author: Administrator
"""

from pandas import read_csv
import matplotlib.pyplot as plt

#filename = 'F:/Python/pima/diabetes.csv'
## names = ['Number of times pregnant', 
##          'Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 
##          'Diastolic blood pressure (mm Hg)',
##          'Triceps skin fold thickness (mm)',
##          '2-Hour serum insulin (mu U/ml)',
##          'Body mass index (weight in kg/(height in m)^2)',
##          'Diabetes pedigree function',
##          'Age (years)',
##          'Class variable (0 or 1)'
##         ]
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#data = read_csv(filename, names=names) # 手动指定头部
data = read_csv('F:/Python/pima/diabetes.csv')
data.hist(figsize = (16,14))
plt.show()

data.plot(kind='density', subplots=True, layout=(3,3),sharex=False,figsize = (16,14))
plt.show()