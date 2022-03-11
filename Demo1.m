% --------------------- A Demo for Matrix Completion  ---------------------
% 
% A simple demo for matrix completion, including the following methods:
%
% -- SVP
% -- SVT
% -- Sp-lp
% -- Sp-lp-new
% -- TNNR 
%
% Author: Hao Liang 
% Last modified by: 22/03/11
%

%% Experiment setup
clc; clear; close all;

% Load figure
image = imread('house.png'); image = image(:,:,1); img = double(image);

% Normalization
xm = min(img(:)); Io = img - xm; img = Io/max(Io(:)); 

% Parameters setting
[nx,ny,~] = size(image);
tol = 1e-4;      % stopping criteria
maxIter = 1000;  % maximum allowable iterations

% Random mask
mask = zeros(nx,ny); samp_rate = 0.4;  % sampling rate 
chosen = randperm(nx*ny,round(samp_rate*nx*ny)); mask(chosen) = 1 ;

% Masked image
mask_image = img.*mask; 

%% SVP
step = 1/samp_rate/sqrt(maxIter); k = 10;
SVP_recon = SVP(mask_image,mask,step,k,maxIter,tol);

%% SVT
tao = sqrt(nx*ny); step = 1.2*samp_rate; 
SVT_recon = SVT(mask_image,mask,tao,step,maxIter,tol);

%% Sp-lp
gamma = 1; p = 0.1;
Sp_lp_recon = Sp_lp(mask_image,mask,gamma,p,maxIter,tol);

%% TNNR-ADMM
beta = 1; rank_r = 1;
TNNR_recon = TNNR_ADMM(mask_image,mask,beta,rank_r,maxIter,tol);

%% Sp-lp-new
gamma = 1; p1 = 0.1; p2 = 0.2;
Sp_lp_new_recon = Sp_lp_new(mask_image,mask,gamma,p1,p2,maxIter,tol);

%% Experimental results
figure; imshow(image,[]); title('Original image','FontSize',15,'FontName','Times New Roman'); 
figure; imshow(mask_image,[]); title(['Masked image (sampling rate = ', num2str(samp_rate),')'],'FontSize',15,'FontName','Times New Roman'); 
figure; imshow(SVP_recon,[]); title('Recovered image by SVP','FontSize',15,'FontName','Times New Roman'); 
figure; imshow(SVT_recon,[]); title('Recovered image by SVT','FontSize',15,'FontName','Times New Roman'); 
figure; imshow(Sp_lp_recon,[]); title('Recovered image by Sp-lp','FontSize',15,'FontName','Times New Roman'); 
figure; imshow(TNNR_recon,[]); title('Recovered image by TNNR-ADMM','FontSize',15,'FontName','Times New Roman'); 
figure; imshow(Sp_lp_new_recon,[]); title('Recovered image by Sp-lp-new','FontSize',15,'FontName','Times New Roman'); 
