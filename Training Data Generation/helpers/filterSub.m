% (C) Copyright 2022                
%     All rights reserved           
%
% Author: Peiyi Zhang, Nov 2022

function [im_max] = filterSub(data, thresh)
sumim = data;
sz=16/5;
tic
im_unif = unif(sumim,[sz sz 0],'rectangular')-unif(sumim,[2*sz 2*sz 0],'rectangular');
toc
loc_max = (im_unif>=.999*maxf(im_unif,[2*sz 2*sz 0],'rectangular'));
im_max = loc_max & (im_unif>thresh(1));

