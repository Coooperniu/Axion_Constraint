import sys
import os  
import numpy as np
import numexpr as ne

#print("This is the name of the program:",
#       sys.argv[1:])
#print("Number of elements including the name of the program:",
#       len(sys.argv))
print("Number of elements excluding the name of the program:",
#      (len(sys.argv)-1))
#print("Argument List:",
#       str(sys.argv))

from multiprocessing import cpu_count

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

import datetime

x = datetime.datetime.now()
print(x)

import time

output_path = 'output'+ time.strftime('-%Y-%m-%d-%H-%M')
print(output_path)
