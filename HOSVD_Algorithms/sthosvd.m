function [G,A] = sthosvd(X,r,order)
% Calculates STHOSVD of tensor X in Tucker form
% 
% Inputs
%   X: original tensor (d modes)
%   r: target rank vector [r1,r2,...,rd]
%   order: processing order vector of modes (can be less than length d)
%
% Outputs
%   G: core tensor (size(G) = r)
%   A: cell array of factor matrices
% 
% Written by Rachel Minster, 2018

G = X; 
sz = getsize(X);
d = length(order);
A = cell(1,d);

for j = 1:d
    k = order(j);
    M = tens2mat(G,k); %matricize kth mode
    
    [U,~,~] = svd(M,'econ'); 
    
    Q = U(:,1:r(k));
    A{j} = Q;   %save factor matrix
    
    sz(k) = r(k);
       
    G = mat2tens(Q'*M,sz,k);  %form new core
end
end