function [vec] = cell2vec(cell)
% cell is N*35*35*4   (number of sample, ROI_X,ROI_Y,view)

samplesNum = size(cell,2);
nv = size(cell{1},3);
vec = [];
for t=1:samplesNum
    features = [];
    for v=1:nv
        features = [features vectorize(cell{t}(:,:,v))];
    end
    vec = [vec;features];
end