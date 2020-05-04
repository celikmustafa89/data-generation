function [samples,fakeView] = generateFeature(representative_tensor,N,nv,NumFeatures,numROIs, sigma)

% vectorize representative tensors
tensors = representative_tensor;
view_vec = {nv};
for k=1:nv
    view_vec{k} = vectorize(tensors{k});
end

% extract related features from different view to vector
features = {NumFeatures};
for f=1:NumFeatures
    values = [];
    for v=1:nv
        values = [values view_vec{v}(f)];
    end
    features{f} = values; 
end

% generating samples
% resource: https://www.mathworks.com/help/stats/mvnrnd.html

for k=1:size(features,2)
    mu = features{k};
    rng(1); % this line is for generating same sample for same values.
    R{k} = mvnrnd(mu,sigma{k},N);
end

% extract samples from features
samples = cell(1,N);
for k=1:N
    view_values = cell(1,nv);
    for l=1:NumFeatures
        for v=1:nv
            view_values{v} = [view_values{v} R{l}(k,v)];
        end        
    end
    samples{k} = view_values;
end

% generate sample matrix
mat = cell(1,N);
for k=1:N
    view_values = cell(1,nv);
    for l=1:nv
        view_values{l} = anti_vectorize(samples{k}{l}',numROIs);
    end
    mat{k} = view_values;
end

% change format of matrix
fakeView = {};
for j=1:nv
    for k=1:N
        fakeView{j}(:,:,k) = mat{k}{j};
    end
end





