function [tensor] = cell2tensor(cell)
% cell   is N*35*35*4   (number of sample, ROI_X,ROI_Y,view)
% tensor is 4*(35*35*N) (view, ROI,ROI,number of sample)

samplesNum = size(cell,2);
nv = size(cell{1},3);
ROI=size(cell{1},1);
tensor = {};
for t=1:samplesNum
    features = [];
    for v=1:nv
        tensor{v}(:,:,t)= cell{t}(:,:,v);
    end
end