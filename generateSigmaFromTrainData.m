function [sigma] = generateSigmaFromTrainData(view,N,nv)

%random sigma values initial method step1

%fake3

for i=1:nv
   V1{i}=vectorize(view{i}); 
end

%Concatenated vectorized views
for i=1:N
    subj1{i}=V1{1}(i,:);
    for j=2:nv
       subj1{i}=[subj1{i};V1{j}(i,:)]; %V{i}:4*595: number of views*number of features
    end
end


% finding covariance of the original features
originalfeatures = zeros(N,nv);
for i=1:N
    originalfeatures(i,:) = subj1{i}(:,k);
end
sigma = cov(originalfeatures);

