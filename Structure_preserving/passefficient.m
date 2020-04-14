function [Q,p] = passefficient(Y,subset)
% Returns Q such that Y \approx QB
%
% Inputs: Y - matrix 
%         subset - subset selection method
%                - 'deim', 'pqr', or 'rrqr'
%
% Outputs: Q such that Y \approx QB
%          p: chosen indices
%
% Written by Arvind K. Saibaba & modified by Rachel Minster, 2018

[Q,~] = qr(Y,0);

% Extract indices and apply the interpolatory projector
p = subsetselection(Q,subset);
Q = Q/Q(p,:);

end