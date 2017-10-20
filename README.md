# Astrarecon

The code in this repository reconstructs topo-tomography datasets collected at ID06 using dark-field X-ray microscopy (DFXRM). The reconstruction is performed using the [ASTRA Toolbox](http://www.astra-toolbox.com/). Still a work in progress.

The ASTRA reconstruction code is optimized to run on a GPU-equipped machine.

Reconstruction steps:

 1. Use `getdata.py` to load, process and store the collected data. The script also returns the sum, projection by projection, of all collected frames. The script runs on Panda2

 2. Reconstruct the sample 3D shape on a GPU machine using `recon.py`

 3. Get a feeling of the reconstruction quality using `makemovie.py`

The scripts `gpurun.py` and `pandarun.py` are designed to shuffle data between Panda2 and the GPU machine.
