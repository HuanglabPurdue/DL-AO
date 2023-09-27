%------ Demo code for simulating aberrated PSFs with measured mirror modes------------
% software requirement: Matlab R2020a or later
%                       Dipimage toolbox 2.8.1 or later
% system requirement:   CPU Intel Core i7
%                       32 GB RAM 
% (C) Copyright 2022              
%     All rights reserved          
%                                 
% Author: Peiyi Zhang, Nov 2022

%% load measured pupil and mirror modes
load('.\ExampleData\SystemPupil.mat');
load('.\ExampleData\MirrorMode.mat');

addpath('.\helpers\');

for f = 1:10
    foldername1=['.\TrainingData\' num2str(f), '\'];                % folders to save portions of training data
    if ~exist(foldername1)                                          %
        mkdir(foldername1);                                         %
    end                                                             %
    imsz = 32;                                                      % input height/width to neural network
    Num = 1000;                                                     % # of PSFs in current folder
    Ntype = size(ExperimentalMirrorMode,3);                         % # of network output
    I = rand(Num, 1).*19000 + 1000;                                 % photon counts per detection plane
    Iratio = rand(Num,1)*0.6+0.9;                                   % photon counts in detection plane 2 : photon counts in detection plane 1
    bg = rand(Num, 1).*299 + 1;                                     % background photon counts per pixel
    zz = -2 + rand(Num, 1).*4;                                      % axial positions of the molecule (unit: micron)
    planedist = 0.576 + rand(Num,1)*0.128 - 0.064;                  % axial distance between two detection plane (unit: micron)
    Nrep = 20;                                                      % # of PSFs sharing the same aberrations
    Ncombo = Num/Nrep;                                              % # of aberration types
    label = repelem(rand(Ncombo,Ntype).*2 - 1, Nrep, 1);            % ground truth label for training neural network

    %% define necessary parameters for plane 1
    tic
    PRstruct1 = [];                                                 %
    PRstruct1.coeff = label;                                        % coefficients of mirror mode add to pupil phase
    PRstruct1.NA = 1.35;                                            % numerical aperture of the objective lens
    PRstruct1.Lambda = 0.68;                                        % center wavelength of the emission band pass filter (unit: micron)
    PRstruct1.RefractiveIndex = 1.406;                              % refractive index of the immersion medium
    PRstruct1.SigmaX = 1.55;                                        % Gaussian filter width in x dimension for OTF rescale, unit is 1/micron in k space, the conversion to real space is 1/(2*pi*SigmaX), unit is micron
    PRstruct1.SigmaY = 1.65;                                        % Gaussian filter width in y dimension for OTF rescale, unit is 1/micron in k space, the conversion to real space is 1/(2*pi*SigmaX), unit is micron
    PRstruct1.Pupil.mag = plane1_PRmag;                             % measured pupil magnitude for detection plane 1
    PRstruct1.Pupil.phase = plane1_PRphase;                         % measured pupil phase for detection plane 1
    %% generate PSF for plane 1
    psfobj1 = PSF_MM(PRstruct1);                                    % create object from PSF_MM class
    psfobj1.Xpos = -3 + 6.*rand(Num,1);                             % x positions of PSFs w.r.t center of plane 1 (unit: pixel)
    psfobj1.Ypos = -3 + 6.*rand(Num,1);                             % y positions of PSFs w.r.t center of plane 1  (unit: pixel)
    psfobj1.Zpos = zz - planedist/2;                                % axial positions of the molecules w.r.t. focus of detection plane 1 (unit: micron) 
    psfobj1.Boxsize = imsz;                                         % input height/width to neural network
    psfobj1.Pixelsize = 0.119;                                      % pixel size on the sample plane (unit: micron)
    psfobj1.PSFsize = 128;                                          % image size used for PSF generation
    psfobj1.nMed = 1.406;                                           % refractive index for sample medium. Set to be the same as immersion media in training data generation.
    psfobj1.precomputeParam();                                      % generate parameters for Fourier space operation
    psfobj1.genPSF1(ExperimentalMirrorMode);                        % generate PSFs
    psfobj1.scalePSF();                                             % generate OTF rescaled PSFs
    psf_plane1 = psfobj1.ScaledPSFs/sum(psfobj1.Pupil.mag(:).^2);   % simulated PSFs
    %% define necessary parameters for plane 2
    PRstruct2 = [];                                                 %
    PRstruct2.coeff = label;                                        % coefficients of mirror mode add to pupil phase 
    PRstruct2.NA = 1.35;                                            % numerical aperture of the objective lens
    PRstruct2.Lambda = 0.68;                                        % center wavelength of the emission band pass filter (unit: micron)
    PRstruct2.RefractiveIndex = 1.406;                              % refractive index of immersion media
    PRstruct2.SigmaX = 1.55;                                        % Gaussian filter width in x dimension for OTF rescale, unit is 1/micron in k space, the conversion to real space is 1/(2*pi*SigmaX), unit is micron
    PRstruct2.SigmaY = 1.65;                                        % Gaussian filter width in y dimension for OTF rescale, unit is 1/micron in k space, the conversion to real space is 1/(2*pi*SigmaX), unit is micron
    PRstruct2.Pupil.mag = plane2_PRmag;                             % measured pupil magnitude for detection plane 2
    PRstruct2.Pupil.phase = plane2_PRphase;                         % measured pupil phase for detection plane 2
    %% generate PSF for plane 2
    psfobj2 = PSF_MM(PRstruct2);                                    % create object from PSF_MM class
    psfobj2.Xpos = psfobj1.Xpos + 3.*rand(Num,1)-1.5;               % x positions of PSFs w.r.t center of plane 2 (unit: pixel)
    psfobj2.Ypos = psfobj1.Ypos + 3.*rand(Num,1)-1.5;               % y positions of PSFs w.r.t center of plane 2 (unit: pixel)
    psfobj2.Zpos = zz + planedist/2;                                % axial positions of the molecules w.r.t. focus of detection plane 2 (unit: micron)
    psfobj2.Boxsize = psfobj1.Boxsize;                              % input height/width to neural network
    psfobj2.PSFsize = psfobj1.PSFsize;                              % image size used for PSF generation
    psfobj2.Pixelsize = psfobj1.Pixelsize;                          % pixel size on the sample plane (unit: micron)
    psfobj2.nMed = psfobj1.nMed;                                    % refractive index of the sample medium    
    psfobj2.precomputeParam();                                      % generate parameters for Fourier space operation
    psfobj2.genPSF1(ExperimentalMirrorMode);                        % generate PSFs
    psfobj2.scalePSF();                                             % generate OTF rescaled PSFs
    psf_plane2 = psfobj2.ScaledPSFs/sum(psfobj2.Pupil.mag(:).^2);   % simulated PSFs
    toc
    %% compose sub-regions with photon, background and Poisson noise
    imsp = zeros(imsz, imsz, 2, Num);
    for i = 1:Num
        imsp(:,:,1,i) = psf_plane1(:,:,i)*I(i)+bg(i);
        imsp(:,:,2,i) = psf_plane2(:,:,i)*I(i).*Iratio(i)+bg(i).*Iratio(i);
    end
    
%     % uncomment the following line if generate training data for Net1
%     [Num_filter, imsp, label] = Net1Filter(imsp(:,:,1,:), imsp(:,:,2,:), label);

    imsp = double(noise(imsp, 'poisson'));
    %% save sub-regions and ground truth label
    save([foldername1 '\label.mat'],'label');
    save([foldername1  '\data.mat'],'imsp', '-v7.3');
end