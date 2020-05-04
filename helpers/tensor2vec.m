function [vec1] = tensor2vec(tensor)
% tensor is 4*(35*35*N)

samplesNum = size(tensor{1},3);
nv = size(tensor,2);
vec1 = [];
for t=1:samplesNum
    features = [];
    for u=1:nv
        features = [features vectorize(tensor{u}(:,:,t))];
    end
    vec1 = [vec1;features];
end