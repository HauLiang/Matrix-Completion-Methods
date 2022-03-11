% --------------- A Demo for Matrix Completion (RGB figure) ---------------
% 
% A simple demo for matrix completion, including the following methods:
%
% -- SVP
% -- SVT
% -- Sp-lp
% -- TNNR 
% -- Sp-lp-new
%
% Author: Hao Liang 
% Last modified by: 22/03/11
%

%% Experiment setup
clc; clear; close all;

% Load RGB figure
image = imread('RGB_figure.jpg');

% Normalize R, G, and B channels, respectively
image_double = double(image);
img_R = image_double(:,:,1); img_G = image_double(:,:,2); img_B = image_double(:,:,3);
xm_R = min(img_R(:)); Io_R = img_R-xm_R; img_R = Io_R/max(Io_R(:));
xm_G = min(img_G(:)); Io_G = img_G-xm_G; img_G = Io_G/max(Io_G(:)); 
xm_B = min(img_B(:)); Io_B = img_B-xm_B; img_B = Io_B/max(Io_B(:));

% Parameters setting
[nx,ny,~] = size(image_double);
tol = 1e-4;      % stopping criteria
maxIter = 1000;  % maximum allowable iterations

% Random mask
mask = zeros(nx,ny); samp_rate = 0.3;  % sampling rate 
chosen = randperm(nx*ny,round(samp_rate*nx*ny)); mask(chosen) = 1 ;

% Masked image
mask_R = img_R.*mask; mask_G = img_G.*mask; mask_B = img_B.*mask;
mask_image = cat(3,mask_R,mask_G,mask_B);

%% SVP
step = 1/samp_rate/sqrt(maxIter); rankk = 10;
SVP_recon_R = SVP(mask_R,mask,step,rankk,maxIter,tol);
SVP_recon_G = SVP(mask_G,mask,step,rankk,maxIter,tol);
SVP_recon_B = SVP(mask_B,mask,step,rankk,maxIter,tol);
SVP_image = cat(3,SVP_recon_R,SVP_recon_G,SVP_recon_B);

%% SVT
tao = sqrt(nx*ny); step = 1.2*samp_rate; 
SVT_recon_R = SVT(mask_R,mask,tao,step,maxIter,tol);
SVT_recon_G = SVT(mask_G,mask,tao,step,maxIter,tol);
SVT_recon_B = SVT(mask_B,mask,tao,step,maxIter,tol);
SVT_image = cat(3,SVT_recon_R,SVT_recon_G,SVT_recon_B);

%% Sp-lp
gamma = 1; p = 0.1;
Sp_lp_recon_R = Sp_lp(mask_R,mask,gamma,p,maxIter,tol);
Sp_lp_recon_G = Sp_lp(mask_G,mask,gamma,p,maxIter,tol);
Sp_lp_recon_B = Sp_lp(mask_B,mask,gamma,p,maxIter,tol);
Sp_lp_image = cat(3,Sp_lp_recon_R,Sp_lp_recon_G,Sp_lp_recon_B);

%% TNNR-ADMM
beta = 1; rank_r = 1;
TNNR_recon_R = TNNR_ADMM(mask_R,mask,beta,rank_r,maxIter,tol);
TNNR_recon_G = TNNR_ADMM(mask_G,mask,beta,rank_r,maxIter,tol);
TNNR_recon_B = TNNR_ADMM(mask_B,mask,beta,rank_r,maxIter,tol);
TNNR_image = cat(3,TNNR_recon_R,TNNR_recon_G,TNNR_recon_B);

%% Sp-lp-new
gamma = 1; p1 = 0.1; p2 = 0.2;
Sp_lp_new_recon_R = Sp_lp_new(mask_R,mask,gamma,p1,p2,maxIter,tol);
Sp_lp_new_recon_G = Sp_lp_new(mask_G,mask,gamma,p1,p2,maxIter,tol);
Sp_lp_new_recon_B = Sp_lp_new(mask_B,mask,gamma,p1,p2,maxIter,tol);
Sp_lp_new_image = cat(3,Sp_lp_new_recon_R,Sp_lp_new_recon_G,Sp_lp_new_recon_B);

%% Experimental results
figure; imshow(image,[]); title('Original image','FontSize',15,'FontName','Times New Roman'); 
figure; imshow(mask_image,[]); title(['Masked image (sampling rate = ', num2str(samp_rate),')'],'FontSize',15,'FontName','Times New Roman'); 
figure; imshow(SVP_image,[]); title('Recovered image by SVP','FontSize',15,'FontName','Times New Roman'); 
figure; imshow(SVT_image,[]); title('Recovered image by SVT','FontSize',15,'FontName','Times New Roman'); 
figure; imshow(Sp_lp_image,[]); title('Recovered image by Sp-lp','FontSize',15,'FontName','Times New Roman'); 
figure; imshow(TNNR_image,[]); title('Recovered image by TNNR-ADMM','FontSize',15,'FontName','Times New Roman'); 
figure; imshow(Sp_lp_new_image,[]); title('Recovered image by Sp-lp-new','FontSize',15,'FontName','Times New Roman'); 
