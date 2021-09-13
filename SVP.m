function SVP_recon = SVP(mask_image,mask,step,k,maxIter,tol)
%
% This code implements the SVP algorithm
%
% More information about SVP can be found in the paper:
%    Meka, Raghu and Jain, Prateek and Dhillon, Inderjit S, 
%    "Guaranteed rank minimization via singular value projection", 
%    arXiv preprint arXiv:0909.5457, 2009.
%
%
% Inputs:
%    mask_image:  sampled image
%    mask:  sampled set
%    step:  step size
%    k:     the maximum allowable rank
%    maxIter:  the maximum allowable iterations
%    tol:   tolerance of convergence criterion
%
% Outputs:
%    SVP_recon: recovered image, obtained by SVP
%
% Author: Hao Liang 
% Last modified by: 21/09/13
%

% Initialization
PICKS = find(mask==1); [nx,ny] = size(mask_image);
X = zeros(nx,ny); 

for iter = 1:maxIter  
    
    % Update Y
    Y = X-step*(mask.*X-mask_image);
    Y(PICKS) = mask_image(PICKS);
    Y = abs(Y);    
   
    % Singular value decomposition 
    [U,S0,V] = svd(Y,'econ'); 
    S1 = diag(S0);   
    S = S1(1:k,1);  %(1:rank_r,1)
 
    % Update X
    Xtemp = X; X = U(:,1:k)*diag(S)*(V(:,1:k))';
    X(PICKS) = mask_image(PICKS); X = abs(X);

    % Stopping criteria
    TOLL = norm(X-Xtemp,'fro')/max(norm(X,'fro'),1);
    if TOLL < tol
       break;
    end 
    
end

SVP_recon = X;

end
