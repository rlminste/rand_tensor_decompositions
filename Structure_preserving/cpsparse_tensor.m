function X = cpsparse_tensor(n,sparsity,gap)
% Constructs synthetic sparse tensor
%
% Inputs
% n: size of tensor (n,n,n)
% sparsity: percentage of nonzeros
% gap: gap between first 10 terms and last terms
%
% Outputs
% X: sparse tensor
%
% - Written by Arvind K. Saibaba, modified by Rachel Minster 2018

addpath('tensorlab')

X = zeros(n,n,n);
for i = 1:10
    x = full(sprand(n,1,sparsity));
    y = full(sprand(n,1,sparsity));
    z = full(sprand(n,1,sparsity));
    X = X + (gap/i^2)*cpdgen({x,y,z});
end


for i = 11:n
    x = full(sprand(n,1,sparsity));
    y = full(sprand(n,1,sparsity));
    z = full(sprand(n,1,sparsity));
    X = X + (1/i^2)*cpdgen({x,y,z});
end

rmpath('tensorlab')
end