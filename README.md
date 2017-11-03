# Astrarecon

Contributors: A.C. Jakobsen, A. Cereser

Technical University of Denmark, Department of Physics, NEXMAP

andcj@fysik.dtu.dk, alcer@fysik.dtu.dk

*** Still a work in progress ***

The code in this repository reconstructs topo-tomography datasets collected at ID06 (ESRF) using dark-field X-ray microscopy (DFXRM). The reconstruction code is based on the [ASTRA Toolbox](http://www.astra-toolbox.com/). The Astrarecon reconstruction approach can be combined with the [Recon3D](https://github.com/albusdemens/Recon3D) one, which returns 3D shape and orientation distribution of the considered grain from the same input data.

The ASTRA reconstruction code here available is designed to run on a GPU-equipped machine.

Reconstruction steps:

 1. Use `getdata.py` to load, process and store the collected frames. The script also sums, projection by projection, the collected frames. The script runs on Panda2

 2. Reconstruct the sample 3D shape on a GPU machine using `recon.py`

 3. Check the reconstruction quality using `makemovie.py`

 The functioning of the script is described in the [Recon3D manual](https://github.com/albusdemens/Recon3D/blob/master/Manual_Recon3D.pdf).

The scripts `gpurun.py` and `pandarun.py` are designed to transfer data between Panda2 and the GPU machine.
