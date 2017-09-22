% This script shows the sinogram for the collected projections, to help deciding
% if there are projections to exclude

clear; close all;

% Load images from Fabio_isolate_blob
A = load('/u/data/alcer/DFXRM_rec/Rec_test_2/Sample2_cleaned.mat');

Data = A.foo;

R_all = zeros(size(Data,1), size(Data,2));

for ii = 1:size(Data,1)
    angle = ii*1.125;
    R_all(ii,:) = squeeze(Data(ii,150,:));
end

figure;  h = pcolor(R_all'); shading flat;
title('Sinogram for the collected projections');
xlabel('Projection number');
ylabel('Pixel');