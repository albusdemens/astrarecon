#!/usr/local/bin/python

# This script synchronizes the local files with the files on the gpu machine
# and runs them

import os
import sys

os.system('rsync -avz /Users/Alberto/Documents/Data_analysis/DFXRM/astrarecon/*.py gpu@130.225.86.15:astra-recon/.')

os.system('ssh gpu@130.225.86.15 \". ~/.bash_profile; cd astra-recon; python %s\"' % (sys.argv[1]))
# os.system('ssh gpu@130.225.86.15 \"cd astra-recon; mpiexec -n %s python %s\"' % (sys.argv[1], sys.argv[2]))
# os.system('ssh andcj@panda2.fysik.dtu.dk \"cd dfxm; sh make_rec_data.sh\"')

os.system('rsync -avz gpu@130.225.86.15:astra-recon/output/ /Users/Alberto/Documents/Data_analysis/DFXRM/astrarecon/output/. ')
