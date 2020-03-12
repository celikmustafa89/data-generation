function [distance, newView]=fitness_function(eval_view,representative_tensor,N,nv,NumFeatures,numROIs,sigma,generation_type)


if strcmp(generation_type,'gen1')
    [~, newView] = generateSampleGA(representative_tensor,N,nv,NumFeatures,numROIs, sigma);
end

if strcmp(generation_type,'gen2')
    [~, newView] = generateSampleGAEachFeature(representative_tensor,N,nv,NumFeatures,numROIs, sigma);
end

distance = cross_distance(eval_view,N,newView,N,nv);


