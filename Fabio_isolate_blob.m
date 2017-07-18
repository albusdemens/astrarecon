% This scipt loads the images collected during a topotomo scan and summed
% so to have one image per projection. Output: cleaned input, with 
% isolated diffraction signal

% Load data file made by Clean_sum_img.py
clear; close all;
addpath('/home/nexmap/alcer/Astra/npy-matlab-master/');
A = load('Fabio_clean.mat');
Fab = A.foo;

% Here we store all positive values (before cleaning)
All_pos = zeros(size(Fab,1), size(Fab,2), size(Fab,3));
% Here we store all clean values
All_clean_temp = zeros(size(Fab,1), size(Fab,2), size(Fab,3));
All_clean = zeros(size(Fab,1), size(Fab,2), size(Fab,3));
mean_mask = zeros(size(Fab,1),1);
for i = 1:size(Fab,1)
    C = Fab(i, :, :);
    [Masked_im,D] = clean_fun(C);
    for j = 1:size(Fab,2)
        for k = 1:size(Fab,3)
            All_clean_temp(i,j,k) = Masked_im(j,k); 
            All_pos(i,j,k) = D(j,k);
        end
    end
    mean_mask(i) = mean(mean(Masked_im));
end
% To take into account of the varying absorption length as the sample
% rotates, we divide the images by the average value
max_mean = max(mean_mask);
for i = 1:size(Fab,1)
    for j = 1:size(Fab,2)
        for k = 1:size(Fab,3)
            All_clean(i,j,k) = All_clean_temp(i,j,k) / mean_mask(i) * max_mean;
        end
    end
end

save('All_clean_images.mat','All_clean');

% Export the cleaned data, which will be loaded by getdata.py
Astra_input = zeros(size(All_clean,2), size(All_clean,1), size(All_clean,3));
for i = 1:size(All_clean,1)
    for j = 1:size(All_clean,2)
        for k = 1:size(All_clean,3)
            Astra_input(j,i,k) = All_clean(i,j,k);
        end
    end
end

% Save to an npy file the input for Astra
%writeNPY(Astra_input,'/u/data/alcer/DFXRM_rec/Rec_test/Astra_input.npy');

% Integrated intensity before and after division by mean
Int_layer = zeros(size(All_clean,1), 3);
for i = 1:size(All_clean,1)
    Int_layer(i,1) = i;
    Int_layer(i,2) = sum(sum(All_clean_temp(i,:,:)));
    Int_layer(i,3) = sum(sum(All_clean(i,:,:)));
end

figure;
scatter(Int_layer(:,1), Int_layer(:,2), '.r');
hold on;
scatter(Int_layer(:,1), Int_layer(:,3), '.b');
title('Integrated intensity before and after normalization')
xlabel('Projection number')
ylabel('Integrated intensity')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function cleans, for each projection, the summed images using frames
% with no diffraction signal
function [Masked_im,D] = clean_fun(C)

    D = zeros(300,300); % Only positive values
    E = zeros(300,300); % Threshold
    F = zeros(300,300); % Thresholded and binarized
    for i = 1:300
        for j = 1:300
            if C(1,i,j) > 0
                D(i,j) = C(1,i,j);
            end
            if C(1,i,j) > 1000
                E(i,j) = C(1,i,j);
                F(i,j) = 1;
            end
        end
    end

    % Recongnize shapes in binarized image
    L = bwlabel(F);
    s = regionprops(L, 'area');
    count_big = 0;
    Counter_big = zeros(size(s,1), 2);
    for i = 1:size(s,1)
        if s(i).Area > 1000
            count_big = count_big + 1;
            Counter_big(count_big,1) = i;
            Counter_big(count_big,2) = s(i).Area;
        end
    end
    Counter_big( ~any(Counter_big,2), : ) = []; 

    % Make a mask, where only the big grains are taken into account
    Mask = zeros(300,300);
    Mask_dil = zeros(300,300);
    Masked_im = zeros(300,300);
    for i = 1:count_big
        for j = 1:300
            for k = 1:300
                if L(j,k) == Counter_big(i,1)
                    Mask(j,k) = 1;
                end
            end
        end
    end

    Mask = imfill(Mask, 'holes');
    se = strel('disk',10);
    Mask_dil = imdilate(Mask,se);

    for j = 1:300
        for k = 1:300
            if Mask_dil(j,k) > 0
                Masked_im(j,k) = D(j,k);
            end
        end
    end

end

%figure; 
%subplot(121);
%imagesc(D); colormap(jet); title('Raw image');
%colorbar;
%subplot(122);
%imagesc(Masked_im); colormap(jet); title('Raw image');
%colorbar;
