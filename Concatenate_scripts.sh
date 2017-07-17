cd /home/nexmap/alcer/DFXRM/Recon3D/ &&
python getdata.py /u/data/andcj/hxrm/Al_april_2017/topotomo/sundaynight topotomo_frelon_far_ /home/nexmap/alcer/DFXRM/bg_refined topotomo_frelon_far_ 256,256 300,300 /u/data/alcer/DFXRM_rec Rec_test 0.785 -3.319 &&
ls /u/data/andcj/hxrm/Al_april_2017/topotomo/sundaynight/ > /u/data/alcer/DFXRM_rec/Rec_test/List_images.txt &&
python image_properties.py &&
cd /home/nexmap/alcer/Astra/ &&
python Clean_sum_img.py &&
matlab -nodesktop -nodisplay -r "Fabio_isolated_blob.m" &&
scp /u/data/alcer/DFXRM_rec/Rec_test/Astra_input.npy gpu@130.225.86.15:/home/gpu/astra_data/April_2017_sundaynight/ &&
python recon.py
