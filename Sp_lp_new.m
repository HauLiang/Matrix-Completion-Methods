function Sp_lp_new_recon = Sp_lp_new(mask_image,mask,gamma,p1,p2,maxIter,tol)
%
% This code implements the Sp-lp-new algorithm
%
% More information about Sp-lp-new can be found in the paper:
%    Nie, Feiping and Wang, Hua and Huang, Heng and Ding, Chris, 
%    "Joint Schatten p-norm and lp-norm robust matrix completion for missing value recovery", 
%    Knowledge and Information Systems, 2015.
%
%
% Inputs:
%    mask_image:  sampled image
%    mask:  sampled set
%    gamma: regularization parameter
%    p1:     the p value of the schatten p-norm 
%    p2:     the p value of the lp-norm 
%    maxIter:  the maximum allowable iterations
%    tol:   tolerance of convergence criterion
%
% Outputs:
%    Sp_lp_new_recon:  recovered image, obtained by Sp-lp-new
%
% Author: Hao Liang 
% Last modified by: 22/03/10
%

% When the image has fewer rows than columns, transpose the image
[nx, ny] = size(mask_image); transpose = 0;
if nx < ny
    transpose = 1;
    mask_image = mask_image';
    mask = mask';
    [nx, ny] = size(mask_image);
end

% Initialization and parameter setting
X = mask_image.*mask;
idx = find(mask==1);     % sampled matrix index
idx1 = find(mask~=1);    % unsampled matrix index
miu = 0.01; rho = 1.2;   % 1<rho<2
Sigma = zeros(nx,ny); Lamda = zeros(nx,ny);
E = zeros(nx,ny); Z = zeros(nx,ny); 

for i = 1:maxIter   
    % Update X
    N = Z-Sigma/miu; M = E+mask_image+Lamda/miu;
    Xtemp = X;
    X(idx) = (N(idx)+M(idx))/2; X(idx1) = N(idx1);
    TOLL = norm(X-Xtemp,'fro')/max(norm(X,'fro'),1);
      
    % Stopping criteria
    if TOLL<=tol
        break;
    end
    
    % Update E
    H = X-mask_image-Lamda/miu;
    for j = 1:length(idx)
        E(idx(j)) = solve_subproblem_11(H(idx(j)), 2/miu, p1);
    end
    
    % Update Z
    G = X + Sigma/miu; [U,S,V] = svd(G,0);
    s = diag(S); 
    for j = 1:length(s)
        s(j) = solve_subproblem_12(s(j),2*gamma/miu, p2);
    end   
    Z = U*diag(s)*V';
    
    % Update Lamda
    Lamda = Lamda + miu*(E-X+mask_image);
    
    % Update Sigma
    Sigma = Sigma + miu*(X-Z);
    
    % Update miu
    miu = miu*rho;
end

if transpose == 1
    X = X.';
end

Sp_lp_new_recon = X;

end


function x = solve_subproblem_11(alpha, lambda, p)

% Solving the subproblem (11), i.e.,
%    min_{x}  (x-alpha)^2 + lambda*|x|^p

a = (lambda*p*(1-p)/2)^(1/(2-p))+eps;
b = 2*a-2*alpha+lambda*p*a^(p-1);
if b < 0
    x = 2*a;
    for i = 1:10
        f = 2*x-2*alpha+lambda*p*x^(p-1);
        g = 2+lambda*p*(p-1)*x^(p-2);
        x = x-f/g;
    end
    sigma1 = x;
    ob1 = x^2-2*x*alpha+lambda*abs(x)^p;
else
    sigma1 = 1;ob1=inf;
end

a = -a;
b = 2*a-2*alpha+lambda*p*abs(a)^(p-1);
if b > 0
    x = 2*a;
    for i = 1:10
        f = 2*x-2*alpha-lambda*p*abs(x)^(p-1);
        g = 2+lambda*p*(p-1)*abs(x)^(p-2);
        x = x-f/g;
    end
    sigma2 = x;
    ob2 = x^2-2*x*alpha+lambda*abs(x)^p;
else
    sigma2 = 1;ob2=inf;
end
sigma_can = [0,sigma1,sigma2];
[~,idx] = min([0,ob1,ob2]);
x = sigma_can(idx);

end


function sigma = solve_subproblem_12(alpha, lambda, p)

% Solving the subproblem (12), i.e.,
%    min_{sigma>=0}  (sigma-a)^2 + lambda*|sigma|^p

a = (lambda*p*(1-p)/2)^(1/(2-p))+eps;
b = 2*a-2*alpha+lambda*p*a^(p-1);
if b < 0
    sigma = 2*a;
    for i = 1:10
        f = 2*sigma-2*alpha+lambda*p*sigma^(p-1);
        g = 2+lambda*p*(p-1)*sigma^(p-2);
        sigma = sigma-f/g;
    end
    ob = sigma^2-2*sigma*alpha+lambda*sigma^p;
    if ob > 0
        sigma = 0;
    end
else
    sigma = 0;
end

end