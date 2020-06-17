function [ind,col] = srrqr(M,k)
%
% Determines the indices for a revealing set of columns for a matrix
%
% Inputs
% M: m x n matrix with m <= n
% k: number of columns
%
% Outputs
% ind: indices of the k columns
% col: the k columns
%
% - Written by Rachel Minster, 2020

% initializing
[m,n] = size(M);
P = 1:n;
[Q,R,P] = qr(M,0);
count_perm = 0;
increase = true;

while (increase)
    A = R(1:k,1:k);
    AinvB = A\R(1:k,k+1:n);
    C = R(k+1:m,k+1:n);
    
    % col norms of C
    C_colnrm = zeros(n-k,1);
    for i = 1:n-k
        C_colnrm(i) = norm(C(:,i),2);
    end
    
    % row norms of Ainv
    [U,S,V] = svd(A);
    Ainv = V*diag(1./diag(S))*U';
    Ainv_rownrm = zeros(k,1);
    for j = 1:k
        Ainv_rownrm(j) = norm(Ainv(j,:),2);
    end
    
    % finding indices
    tmp = C_colnrm*Ainv_rownrm';
    F = AinvB.^2+tmp.^2;
    [p,q] = find(F>1,1);
    if (isempty(p))
        increase = false;
    else
        count_perm = count_perm+1;
        R(:,[p q+k]) = R(:,[q+k p]);
        P([p q+k]) = P(q+k p]);
        [Q,R] = qr(R,0);
    end
end

inv_nrm = norm(inv(R(1:k,1:k)));
residual_nrm = norm(R(k+1:m,k+1:n));

ind = sort(P(1:k),'ascend');
col = M(:,ind);
end