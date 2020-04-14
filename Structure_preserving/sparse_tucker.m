function [G,U,inds] = sparse_tucker(X,r,p,modes)
% Computes Structure-Preserving STHOSVD 
%
% Inputs
%   X: original tensor (d modes)
%   r: target rank vector [r1,...,rd]
%   p: oversampling parameter (usually in the 5-20 range)
%   modes: processing order vector of modes to compress
%
% Outputs
%   G: core tensor of size r
%   U: cell array of factor matrices
%   inds: cell array of chosen indices from X that form core tensor
%
% Uses Tensor Toolbox
%
% - Written by Rachel Minster, 2018

sz = size(X);
m = length(sz);
d = length(modes);
G = X;
U = cell(1,d);
inds = cell(1,d);

for i = 1:d
    j = modes(i);
    M = sptenmat(G,j);
    num = max(sz(1:end ~= j));
    cols = size(M,2);
    Y = zeros(size(M,1),r(j)+p);
    
    % multiply M*Omega
    for k = 1:num
    	Omega = randn(cols/num,r(j)+p); 
        subs = M.subs;
        flag = isempty(subs); %1 if empty, 0 if not
            if flag == 0
                s1 = subs(:,1);
                s2 = subs(:,2);
                
                %find nonzero elements in given chunk of M
                I = find(s2 >= (k-1)*(cols/num)+1 & s2 <= k*(cols/num));   
                vals = M.vals; 
                v = vals(I);  
                
                %forms small sparse matrix with chosen indices/values
                A = sparse(s1(I),s2(I)-(k-1)*cols/num,v,size(M,1),cols/num);  
                Y = Y + A*Omega; 
            end
    end
    [Q,inds{j}] = passefficient(Y,'rrqr');
    
    % save factor matrix
    U{j} = Q;
    sz(j) = r(j)+p;
    
    idx = repmat({':'},m,1);
    idx{j} = inds{j};
    
    % form core tensor
    G = G(idx{:});    
end
end

