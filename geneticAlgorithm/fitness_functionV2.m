function [distance, newView]=fitness_functionV2(eval_view,representative_tensor,N,nv,NumFeatures,numROIs,sigma,generation_type)

[~, newView] = generateFeature(representative_tensor,N,nv,NumFeatures,numROIs, sigma);

distance = cross_distance(eval_view,N,newView,N,nv);


