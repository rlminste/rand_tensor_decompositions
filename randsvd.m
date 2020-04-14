function [U,S,V] = randsvd(A, k, maxiter,Omega)
% Computes a randomized SVD of A
%
% Inputs
%   A: matrix 
%   k: target rank
%   maxiter: number of subspace iterations
%   Omega: Random matrix size (size(A,2) x k+p) with 
%           p some oversampling parameter
%
% Outputs
%   U,S,V such that A \approx USV'
%
% Written by Rachel Minster 2018

% Initial iteration
Y= A*Omega;
[Q,~]=qr(Y,0);     

% Subsequent iterations
for j = 1:maxiter
    Y = A'*Q;       
    [Q,~]=qr(Y,0);
    Y = A*Q;
    [Q,~]=qr(Y,0);
end

% Low rank approximation
B = Q'*A;
[U,S,V] = svd(B,'econ');
U = Q*(U(:,1:k));

% Compress
V = V(:,1:k);
S = S(1:k,1:k);

end