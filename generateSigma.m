function [sigma] = generateSigma(type,nv)

sigma = [];

if strcmp(type,'fake1')
    sigma = ones(nv);
end

if strcmp(type,'fake2')
    rng(1)
    ii = rand(nv);
    sigma = ii*ii.';
end

