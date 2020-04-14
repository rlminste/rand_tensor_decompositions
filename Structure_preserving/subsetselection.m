function [p,err] = subsetselection(V, method)
% Performs subset selection
%
% Inputs: V - matrix 
%         method - subset selection method ('deim','pqr', or 'rrqr')
% 
% Outputs: p - chosen indices
%          err - approximation error
%
% Written by Arvind K. Saibaba, 2018
    
    [~,k] = size(V);
    if strcmp(method,'deim')
        p = deim(V);
        err = norm(inv(V(p,:)));
    elseif strcmp(method, 'pqr')
        [~,~,p] = qr(V',0);
        p = p(1:k); err = norm(inv(V(p,:)));
    elseif strcmp(method, 'rrqr')
        [p,~] = srrqr(V',k);    err = norm(inv(V(p,:)));
    end

end
