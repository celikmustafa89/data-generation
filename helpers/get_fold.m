function [train_class1,test_class1,train_class2,test_class2]=get_fold(class1_tensor_folds, class2_tensor_folds, totalFold,foldNum)


%%
% seperate test samples
test_class1 = class1_tensor_folds{foldNum};
test_class2 = class2_tensor_folds{foldNum};

% get train set
train_class1={};
train_class2={};
for j=1:totalFold
    if j ~= foldNum
        train_class1 = [train_class1 class1_tensor_folds{j}];
        train_class2 = [train_class2 class2_tensor_folds{j}];
    end
end