# Anders C. Jakobsen, July 2017
# DTU Fysik

# Script to manage the reconstruction code on the GPU machine at DTU Fysik and
# on Panda2

#!/usr/local/bin/python

import os
import sys

os.system('rsync -avz /Users/andcj/astra-recon/*.py andcj@panda2.fysik.dtu.dk:/u/data/andcj/astra-recon/.')
os.system('rsync -avz /Users/andcj/astra-recon/lib/*.py andcj@panda2.fysik.dtu.dk:/u/data/andcj/astra-recon/lib/.')

os.system('ssh andcj@panda2.fysik.dtu.dk \"cd /u/data/andcj/astra-recon; python %s\"' % (sys.argv[1]))
# os.system('ssh gpu@gpureco \"cd astra-recon; mpiexec -n %s python %s\"' % (sys.argv[1], sys.argv[2]))
# os.system('ssh andcj@panda2.fysik.dtu.dk \"cd dfxm; sh make_rec_data.sh\"')

os.system('rsync -avz andcj@panda2.fysik.dtu.dk:/u/data/andcj/astra-recon/output/ /Users/andcj/astra-recon/output/. ')
