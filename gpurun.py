# Anders C. Jakobsen, July 2017
# DTU Fysik

# Script to manage the reconstruction on the GPU machine available at DTU Fysik

#!/usr/local/bin/python

import os
import sys

os.system('rsync -avz /Users/andcj/astra-recon/*.py gpu@130.225.86.15:astra-recon/.')

os.system('ssh gpu@130.225.86.15 \". ~/.bash_profile; cd astra-recon; python %s\"' % (sys.argv[1]))
# os.system('ssh gpu@130.225.86.15 \"cd astra-recon; mpiexec -n %s python %s\"' % (sys.argv[1], sys.argv[2]))
# os.system('ssh andcj@panda2.fysik.dtu.dk \"cd dfxm; sh make_rec_data.sh\"')

os.system('rsync -avz gpu@130.225.86.15:astra-recon/output/ /Users/andcj/astra-recon/output/. ')
