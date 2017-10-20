# Astrarecon

Reconstructs the topo-tomography dataset collected at ID06 using dark-field X-ray microscopy (DFXRM). The reconstruction is performed using the [ASTRA Toolbox](http://www.astra-toolbox.com/). Still a work in progress.

The ASTRA reconstruction code is designed to run on a GPU-equipped machine.

Reconstruction steps:

 1. Load, process and store the collected data using `getdata.py`. The script also sums, for each projection, all collected images. The script runs on Panda2.

 2. Reconstruct the sample shape on a GPU machine using `recon.py`

 3. Get a feeling of the reconstruction quality using `makemovie.py`

The scripts `gpurun.py` and `pandarun.py` are designed to copy data between Panda2 and the GPU machine.
