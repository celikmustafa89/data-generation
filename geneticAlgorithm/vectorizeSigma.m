function [vector] = vectorizeSigma(mat)

At = mat.';
m  = (1:size(At,1)).' >= (1:size(At,2));
vector  = At(m);