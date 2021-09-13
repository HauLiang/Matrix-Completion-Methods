function SVT_recon = SVT(mask_image,mask,tao,step,maxIter,tol)
%
% This code implements the SVT algorithm
%
% More information about SVT can be found in the paper:
%    Cai, Jian-Feng and Cand¨¨s, Emmanuel J and Shen, Zuowei, 
%    "A singular value thresholding algorithm for matrix completion", 
%    SIAM Journal on optimization, 2010.
%
%
% Inputs:
%    mask_image:  sampled image
%    mask:  sampled set
%    tao:   threshold parameter
%    step:  step size
%    maxIter:  maximum allowable iterations
%    tol:   tolerance of convergence criterion
%
% Outputs:
%    SVT_recon:  recovered image, obtained by SVT
%
% Author: Hao Liang 
% Last modified by: 21/09/13
%

% Initialization
X = mask_image;
Y = zeros(size(mask_image));

for i = 1:maxIter
    
    % Singular value decomposition to update X
    [U,S,V] = svd(Y,'econ'); 
    S = sign(S).*max(abs(S)-tao,0);
    XTemp = X; X = U*S*V';
    
    % Update Y
    Y = Y+step*mask.*(mask_image-X);
    Y = mask.*Y;
    
    % Stopping criteria
    TOLL = norm(mask.*(XTemp-X),'fro')/norm(mask.*X,'fro');
    if TOLL<tol
        break;
    end
    
end

SVT_recon = X;

end