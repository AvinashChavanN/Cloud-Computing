#!/usr/bin/python

import sys
import csv
import time
from array import *

threshold =50
month =2

Time_Before_Upload = time.time()
for line in sys.stdin:
    line = line.strip()
    array = line.split(',')
    counter = 3
    first_item = array[counter]
    second_item = array[counter+1]
    third_item = array[counter+2]
    try:
        if first_item =='mm':
            counter = 3
        else:
            if second_item == 'mm':
                counter = 4
            else:
                if third_item =='mm':
                    counter = 5
        value1 = float(array[counter+month])
        if value1 < threshold and value1 >0:
            print '%s\t%s' % ('below',line)
        if value1 >=threshold:
            print '%s\t%s' % ('above',line)
    except ValueError:
        continue