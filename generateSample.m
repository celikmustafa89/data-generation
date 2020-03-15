function [samples,fakeView]=generateSample(view,representative_tensor,N,nv,NumFeatures,numROIs,sigmaType)

% vectorize representative tensors
tensors = representative_tensor;
for k=1:nv
    view_vec{k} = vectorize(tensors{k});
end

% extract related features from different view to vector
for f=1:NumFeatures
    values = [];
    for v=1:nv
        values = [values view_vec{v}(f)];
    end
    features{f} = values; 
end

% generating samples
% resource: https://www.mathworks.com/help/stats/mvnrnd.html
for i=1:nv
   V1{i}=vectorize(view{i}); 
end

%Concatenated vectorized views
for i=1:N
    subjects{i}=V1{1}(i,:);
    for j=2:nv
       subjects{i}=[subjects{i};V1{j}(i,:)]; %V{i}:4*595: number of views*number of features
    end
end

for k=1:size(features,2)
    mu = features{k};
    
    % finding covariance of the original features
    originalfeatures =zeros(N,nv);
    for i=1:N
        originalfeatures(i,:) = subjects{i}(:,k);
    end
    
    
    if strcmp(sigmaType,'cov')
        sigma = cov(originalfeatures);
    else
        sigma = generateSigma(sigmaType,nv);
    end
    R{k} = mvnrnd(mu,sigma,N);
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
fakeView={};
for j=1:nv
    for k=1:N
        fakeView{j}(:,:,k) = mat{k}{j};
    end
end





