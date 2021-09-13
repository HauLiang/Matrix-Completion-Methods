function TNNR_recon = TNNR_ADMM(mask_image,mask,beta,r,maxIter,tol)
%
% This code implements the TNNR-ADMM algorithm
%
% More information about TNNR-ADMM can be found in the paper:
%    Hu, Yao and Zhang, Debing and Ye, Jieping and Li, Xuelong and He, Xiaofei, 
%    "Fast and accurate matrix completion via truncated nuclear norm regularization", 
%    IEEE transactions on pattern analysis and machine intelligence, 2012.
%
%
% Inputs:
%    mask_image:  sampled image
%    mask:  sampled set
%    beta:  penalty parameter
%    r:  truncated parameter, focus on minimizing the sum of the smallest min(nx,ny)-r singularvalues
%    maxIter:  the maximum allowable iterations
%    tol:   tolerance of convergence criterion
%
% Outputs:
%    TNNR_recon:  recovered image, obtained by TNNR-ADMM
%
% Author: Hao Liang 
% Last modified by: 21/09/13
%

% Initialization
X = mask_image; Y = X; W = X;
PICKS = find(mask==1);

for iter = 1:maxIter   
    
    % Singular value decomposition to X
    [U,S,V] = svd(W-Y/beta);
    S = sign(S).*max(abs(S)-1/beta,0);
    Xtemp = X; X = U*S*V';
  
    % Update Al and Bl, respectively
    [U1,~,V1] = svd(W,'econ');
    Al = (U1(:,1:r)).';
    Bl = (V1(:,1:r )).';
    
    % Update W
    W = X+(1/beta)*(Y+Al.'*Bl); W(PICKS) = mask_image(PICKS);
    
    % Update Y
    Y = Y+beta*(X-W);
    
    % Stopping criteria
    TOLL = norm(X-Xtemp,'fro')/norm(X,'fro');
    if TOLL<=tol
        break;
    end
    
end

TNNR_recon = X;

end

