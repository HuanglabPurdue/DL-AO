% (C) Copyright 2022                
%     All rights reserved           
%
% Author: Peiyi Zhang, Nov 2022

function [Num_filter, imsp, label] = Net1Filter(img_plane1, img_plane2, aber)
% This function is used for generating training data for Net1
% Select PSFs detectable by segmentation algorithm for training
imsz = size(img_plane1,1);
[im1_max] = filterSub(squeeze(img_plane1), 20);
[im2_max] = filterSub(squeeze(img_plane2), 20);
im_max = im1_max + im2_max;
mask = double(squeeze(max(max(im_max,[],1),[],2)));
psfid = find(mask);
Num_filter = length(psfid);
img_plane1 = img_plane1(:,:,psfid); 
img_plane2 = img_plane2(:,:,psfid); 
imsp = cat(3, reshape(noise(img_plane1,'poisson'),[imsz, imsz, 1,Num_filter]), reshape(noise(img_plane2,'poisson'),[imsz, imsz, 1,Num_filter]));
label = aber(psfid,:);