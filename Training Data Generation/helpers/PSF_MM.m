% (C) Copyright 2022                
%     All rights reserved           
%
% Author: Peiyi Zhang, May 2022


classdef PSF_MM < handle
    % PSF_MM class for generating PSFs from a pupil function
    %   create object: obj = PSF_MM(PRstruct)
    %   input required: PRstruct - define necessary parameters for a PSF model
    %
    % This implementation is modified from PSF_zernike class provided by Sheng Liu. 
    % The complete toolbox can be accessed 
    % https://github.com/HuanglabPurdue/smNet/tree/master/
    %
    % PSF_MM Properties (Input):
    %   Boxsize - 
    %   PRstruct - 
    %   Pixelsize - 
    %   PSFsize - 
    %   Xpos - 
    %   Ypos - 
    %   Zpos - 
    %   
    % PSF_MM Properties (Output):
    %   PSFs - 
    %   ScaledPSFs - 
    %     
    % PSF_MM Methods:
    %   precomputeParam - generate images for k space operation
    %   genPSF1 - generate PSFs from the given pupil function
    % 
    %   see also OTFrescale
    properties
        % PRstruct - define necessary parameters for a PSF model
        %   NA
        %   Lambda
        %   RefractiveIndex
        %   Pupil: phase retrieved pupil function
        %           phase: phase image
        %           mag: magnitude image
        %   coeff: coefficient of mirror mode representing the pupil phase
        %   SigmaX: sigmax of Gaussian filter for OTF rescale, unit is 1/micron in k space, the conversion to real space is 1/(2*pi*SigmaX), unit is micron
        %   SigmaY: sigmay of Gaussian filter for OTF rescale, unit is 1/micron in k space, the conversion to real space is 1/(2*pi*SigmaY), unit is micron
        PRstruct;
        Xpos;% x positions of simulated emitters, a vector of N elements, unit is pixel
        Ypos;% y positions of simulated emitters, a vector of N elements, unit is pixel
        Zpos;% z positions of simulated emitters, a vector of N elements, unit is micron
        nMed;% refractive index of the sample medium
        PSFsize; % image size of the pupil function 
        Boxsize; % image size of out put PSF
        Pixelsize;% pixel size at sample plane, unit is micron
    end
    
    properties (SetAccess = private, GetAccess = private)
        % precompute images for k space operation
        Zo;% r coordinates of out put PSF, it's a image of PSFsize x PSFsize
        k_r;% k_r coordinates of out put OTF, it's a image of PSFsize x PSFsize
        k_z;% k_z coordinates of out put OTF, it's a image of PSFsize x PSFsize
        Phi;% phi coordinates out put PSF, it's a image of PSFsize x PSFsize
        NA_constrain;% a circle function defining the limit of k_r, it's a image of PSFsize x PSFsize
        Cos1;% cos(theta1), theta1 is the angle of between the k vector and the optical axis in the sample medium
        Cos3;% cos(theta3), theta3 is the angle of between the k vector and the optical axis in the immersion medium
    end
    
    properties (SetAccess = private, GetAccess = public)
        % PSFs - out put PSFs from Fourier transform of the pupil function,
        % it's a 3D matrix of Boxsize x Boxsize x N, N is the number of
        % elements in Xpos.
        PSFs;
        ScaledPSFs;
        % Pupil - pupil function generated from a set of mirror modes
        %           phase: phase image of PSFsize x PSFsize
        %           mag: magnitude image of PSFsize x PSFsize
        Pupil;
    end
    
    methods
        function obj=PSF_MM(PRstruct)
            obj.PRstruct=PRstruct;
        end
        
        function precomputeParam(obj)
            % precomputeParam - generate images for k space operation, and saved in
            % precomputed parameters.
            [X,Y]=meshgrid(-obj.PSFsize/2:obj.PSFsize/2-1,-obj.PSFsize/2:obj.PSFsize/2-1);
            obj.Zo=sqrt(X.^2+Y.^2);
            scale=obj.PSFsize*obj.Pixelsize;
            obj.k_r=obj.Zo./scale;
            obj.Phi=atan2(Y,X);
            n=obj.PRstruct.RefractiveIndex;
            Freq_max=obj.PRstruct.NA/obj.PRstruct.Lambda;
            obj.NA_constrain=obj.k_r<Freq_max;
            obj.k_z=sqrt((n/obj.PRstruct.Lambda)^2-obj.k_r.^2).*obj.NA_constrain;
            sin_theta3=obj.k_r.*obj.PRstruct.Lambda./n;
            sin_theta1=n./obj.nMed.*sin_theta3;
            
            obj.Cos1=sqrt(1-sin_theta1.^2);
            obj.Cos3=sqrt(1-sin_theta3.^2);
        end
        
        
        function genPSF1(obj, ZM)
            N_ceff = size(obj.PRstruct.coeff, 2);
            ceffp=obj.PRstruct.coeff;
            N=numel(obj.Xpos);
            R=obj.PSFsize;
            Ri=obj.Boxsize;
            psfs=zeros(Ri,Ri,N);
            
            % compute pupil_mag matrix with mirror modes
            pupil_mag = obj.PRstruct.Pupil.mag./max(obj.PRstruct.Pupil.mag(:));
            mask = pupil_mag>0;
           
            for ii = 1:N
                pupil_phase = angle(obj.PRstruct.Pupil.phase).*mask;
                for k = 1:N_ceff
                    pupil_phase = pupil_phase + ZM(:, :, k) .* ceffp(ii,k);
                end
                shiftphase=-obj.k_r.*cos(obj.Phi).*obj.Xpos(ii).*obj.Pixelsize-obj.k_r.*sin(obj.Phi).*obj.Ypos(ii).*obj.Pixelsize;
                shiftphaseE=exp(-1i.*2.*pi.*shiftphase);
                defocusphaseE=exp(2.*pi.*1i.*obj.Zpos(ii).*obj.k_z);
                pupil_complex=pupil_mag.*exp(pupil_phase.*1i).*shiftphaseE.*double(defocusphaseE);
                psfA=abs(fftshift(fft2(pupil_complex)));
                Fig2=psfA.^2;
                realsize0=floor(Ri/2);
                realsize1=ceil(Ri/2);
                startx=-realsize0+R/2+1;endx=realsize1+R/2;
                starty=-realsize0+R/2+1;endy=realsize1+R/2;
                psfs(:,:,ii)=Fig2(startx:endx,starty:endy)./R^2;
            end
            obj.PSFs=psfs;
            obj.Pupil.mag = pupil_mag;
            obj.Pupil.phase = [];
        end
        

        function scalePSF(obj)
            % scalePSF - generate OTF rescaled PSFs
            %   It operates 'PSFs' using the OTFrescale class. The OTF
            %   rescale acts as a 2D Gaussian filter, the resulting PSFs
            %   are smoother than the orignal PSFs.
            %
            %   see also OTFrescale
            otfobj=OTFrescale;
            otfobj.SigmaX=obj.PRstruct.SigmaX;
            otfobj.SigmaY=obj.PRstruct.SigmaY;
            otfobj.Pixelsize=obj.Pixelsize;
            otfobj.PSFs=obj.PSFs;
            otfobj.scaleRspace();
            obj.ScaledPSFs=otfobj.Modpsfs;
        end
    end
    
end

