function [features, labels] = combineVectors(vec1,class1,vec2,class2)

% combine two different populations in same train_X and train_y array
features = [];
labels = [];
for t=1:size(vec1,1)
    features = [features ; vec1(t,:)];
    labels = [labels ; class1];
end
for m=size(vec1,1)+1:size(vec1,1)+size(vec2,1)
    features = [features ; vec2(m-size(vec1,1),:)];
    labels = [labels ; class2];
end