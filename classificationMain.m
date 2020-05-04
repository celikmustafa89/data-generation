%load data
files = dir('./data/Data_77subjects/LH77subjects');
for i=1:77
    a=load(strcat("./data/Data_77subjects/LH77subjects/subject"+i+".mat"));
    trainTensor{i} = a.A;
end
trainy = load("./data/Data_77subjects/labels77.mat");
trainy = trainy.labels;

% get details of data
[~,ROIs,nv] = size(trainTensor{1});

% load dataset in different tensors.
X_AD_tensors = {};
for i=1:35
    X_AD_tensors{i}=trainTensor{i};
end
X_MCI_tensors = {};
for i=43:77
    X_MCI_tensors{i-42}=trainTensor{i};
end


% create folds for cross-validation
fold_k=5;
[~,N]=size(X_AD_tensors);
fold_size=N/fold_k;
for i=1:N
    foldnum = fix((i-1)/fold_size) + 1;
    indis = rem(i-1,fold_size)+1;
   
    X_AD_tensor_folds{foldnum}{indis} = X_AD_tensors{i};
    X_MCI_tensor_folds{foldnum}{indis} = X_MCI_tensors{i};
end

% remove this deprecated variable
numIteration = 100;
%for i=1:fold_k
for i=1:1
    % get test set
    datasetTest_AD = X_AD_tensor_folds{i};
    datasetTest_MCI = X_MCI_tensor_folds{i};
    
    % get train set
    t_index = 1;
    datasetTrain_AD={};
    datasetTrain_MCI={};
    for j=1:fold_k
        if j ~= i
            for k=1:fold_size
                for v=1:nv
                    datasetTrain_AD{v}(:,:,t_index)  = X_AD_tensor_folds{fold_k}{k}(:,:,v);
                    datasetTrain_MCI{v}(:,:,t_index) = X_MCI_tensor_folds{fold_k}{k}(:,:,v);
                end
                t_index = t_index + 1;
            end
        end
    end
    
    % genetic algorithm and newViews of generated samples
    trainN=size(datasetTrain_AD{1},3);
    [LH_train_NumFeatures,LH_train_Frob_dist,train_representative_tensor,LH_train_netNorm_CBT] = netNorm_func(datasetTrain_AD,nv,trainN,ROIs);
    [best_sigmasAD, sigma_distances, newViews_AD] = geneticAlgorithm(datasetTrain_AD,numIteration,trainN,nv,ROIs,train_representative_tensor,LH_train_NumFeatures,'half','version1','gen1');
    
    trainN=size(datasetTrain_MCI{1},3);
    [LH_train_NumFeatures,LH_train_Frob_dist,train_representative_tensor,LH_train_netNorm_CBT] = netNorm_func(datasetTrain_MCI,nv,trainN,ROIs);
    [best_sigmasMCI, sigma_distances, newViews_MCI] = geneticAlgorithm(datasetTrain_MCI,numIteration,trainN,nv,ROIs,train_representative_tensor,LH_train_NumFeatures,'half','version1','gen1');

    % vectorize features from generated samples
    trainVec_AD = [];
    for t=1:trainN
        features = [];
        for u=1:nv
            features = [features vectorize(newViews_AD{u}(:,:,t))];
        end
        trainVec_AD = [trainVec_AD;features];
    end
    trainVec_MCI = [];
    for t=1:trainN
        features = [];
        for u=1:nv
            features = [features vectorize(newViews_MCI{u}(:,:,t))];
        end
        trainVec_MCI = [trainVec_MCI;features];
    end
    
    % vectorize features for test samples
    [~,testN]=size(datasetTest_AD);
    testVec_AD = [];
    for t=1:testN
        features = [];
        for u=1:nv
            features = [features vectorize(datasetTest_AD{t}(:,:,u))];
        end
        testVec_AD = [testVec_AD;features];
    end
    [~,testN]=size(datasetTest_MCI);
    testVec_MCI = [];
    for t=1:testN
        features = [];
        for u=1:nv
            features = [features vectorize(datasetTest_MCI{t}(:,:,u))];
        end
        testVec_MCI = [testVec_MCI;features];
    end
    
    % combine two different populations(AD and MCI) in same train array
    X = [];
    y = [];
    for m=1:length(trainVec_AD)
        X = [X ; trainVec_AD(m)];
        y = [y ; 1];
    end
    for m=length(trainVec_AD)+1:length(trainVec_AD)+length(trainVec_MCI)
        X = [X ; trainVec_MCI(m-length(trainVec_AD))];
        y = [y ; 0];
    end
    
    % combine two different populations in same test array
    testX = [];
    testy = [];
    for m=1:length(testVec_AD)
        testX = [testX ; testVec_AD(m)];
        testy = [testy ; 1];
    end
    for m=length(testVec_AD)+1:length(testVec_AD)+length(testVec_MCI)
        testX = [testX ; testVec_MCI(m-length(testVec_AD))];
        testy = [testy ; 0];
    end
    
    % train model
    SVMModel = fitcsvm(X,y);
    
    % test model
    [label,score] = predict(SVMModel,testX);
    models{i} =SVMModel;
    labels{i} = label;
    scores{i} = score;
    actualLabels{i} = testy;
    
    result = [];
    for r=1: length(label)
        result =[result;label(r) testy(r)];
    end

    perf = classperf(testy,label);
    perf.CorrectRate
    perf.Specificity
    perf.Sensitivity
end

avg_accu1=0;
avg_spec1=0;
avg_sens1=0;
for i=1:fold_k
    perf = classperf(actualLabels{i},labels{i});
    accu{i} = perf.CorrectRate;
    spec{i} = perf.Specificity;
    sens{i} = perf.Sensitivity;
    avg_accu1 = avg_accu1 + perf.CorrectRate;
    avg_spec1 = avg_spec1 + perf.Specificity;
    avg_sens1 = avg_sens1 + perf.Sensitivity;
end
avg_accu1 = avg_accu1/fold_k;
avg_spec1 = avg_spec1/fold_k;
avg_sens1 = avg_sens1/fold_k;




ind=5;
datasetTest_AD = X_AD_tensor_folds{ind};
datasetTest_MCI = X_MCI_tensor_folds{ind};
    
% vectorize features for test samples
[~,testN]=size(datasetTest_AD);
testVec_AD = [];
for t=1:testN
    features = [];
    for u=1:nv
        features = [features vectorize(datasetTest_AD{t}(:,:,u))];
    end
    testVec_AD = [testVec_AD;features];
end
[~,testN]=size(datasetTest_MCI);
testVec_MCI = [];
for t=1:testN
    features = [];
    for u=1:nv
        features = [features vectorize(datasetTest_MCI{t}(:,:,u))];
    end
    testVec_MCI = [testVec_MCI;features];
end
    

% combine two different populations in same test array
testX = [];
testy = [];
for m=1:length(testVec_AD)
    testX = [testX ; testVec_AD(m)];
    testy = [testy ; 1];
end
for m=length(testVec_AD)+1:length(testVec_AD)+length(testVec_MCI)
    testX = [testX ; testVec_MCI(m-length(testVec_AD))];
    testy = [testy ; 0];
end
[label,score] = predict(models{ind},testX);
labelss{ind} = label;
actualLabelss{ind} = testy;




[~,traN]=size(trainTensor);
traVec = [];
for t=1:traN
    features = [];
    for u=1:nv
        features = [features vectorize(trainTensor{t}(:,:,u))];
    end
    traVec = [traVec;features];
end
tray = [];
for m=1:traN
    if m < 42
        tray(m) = 1;
    else
        tray(m) = 0;
    end
end
X=traVec;
y=tray;


SVMModel = fitcsvm(X,y,'Standardize',true,'ClassNames',{'1','0'});
CVSVMModel = crossval(SVMModel,'KFold',5);
kfoldLoss(CVSVMModel)
res = kfoldPredict(CVSVMModel);

res1=[];
for i=1:length(res)
    if strcmp(res(1), '1')
        res1 = [res1; 1];
    else
        res1 = [res1; 0];
    end
end

perf = classperf(y,res);
    accu{i} = perf.CorrectRate;
    spec{i} = perf.Specificity;
    sens{i} = perf.Sensitivity;
    avg_accu1 = avg_accu1 + perf.CorrectRate;
    avg_spec1 = avg_spec1 + perf.Specificity;
    avg_sens1 = avg_sens1 + perf.Sensitivity;

% load dataset in different tensors.
X_AD_tensors = {};
for i=1:35
    X_AD_tensors{i}=trainTensor{i};
end
X_MCI_tensors = {};
for i=43:77
    X_MCI_tensors{i-42}=trainTensor{i};
end



%%%%%%%%%%%%
for i=1:fold_k
    % get test set
    datasetTest_AD = X_AD_tensor_folds{i};
    datasetTest_MCI = X_MCI_tensor_folds{i};
    
    % get train set
    t_index = 1;
    datasetTrain_AD={};
    datasetTrain_MCI={};
    for j=1:fold_k
        if j ~= i
            for k=1:fold_size
                for v=1:nv
                    datasetTrain_AD{v}(:,:,t_index)  = X_AD_tensor_folds{fold_k}{k}(:,:,v);
                    datasetTrain_MCI{v}(:,:,t_index) = X_MCI_tensor_folds{fold_k}{k}(:,:,v);
                end
                t_index = t_index + 1;
            end
        end
    end
    
 
    % vectorize features for test samples
    trN=size(datasetTrain_AD{1},3);
    trVec_AD = [];
    for t=1:trN
        features = [];
        for u=1:nv
            features = [features vectorize(datasetTrain_AD{u}(:,:,t))];
        end
        trVec_AD = [trVec_AD;features];
    end
    trN=size(datasetTest_MCI{1},3);
    trVec_MCI = [];
    for t=1:trN
        features = [];
        for u=1:nv
            features = [features vectorize(datasetTrain_MCI{u}(:,:,t))];
        end
        trVec_MCI = [trVec_MCI;features];
    end
    
    
    % vectorize features for test samples
    [~,testN]=size(datasetTest_AD);
    testVec_AD = [];
    for t=1:testN
        features = [];
        for u=1:nv
            features = [features vectorize(datasetTest_AD{t}(:,:,u))];
        end
        testVec_AD = [testVec_AD;features];
    end
    [~,testN]=size(datasetTest_MCI);
    testVec_MCI = [];
    for t=1:testN
        features = [];
        for u=1:nv
            features = [features vectorize(datasetTest_MCI{t}(:,:,u))];
        end
        testVec_MCI = [testVec_MCI;features];
    end
    
    % combine two different populations(AD and MCI) in same train array
    X = [];
    y = [];
    for m=1:length(trVec_AD)
        X = [X ; trVec_AD(m)];
        y = [y ; 1];
    end
    for m=length(trVec_AD)+1:length(trVec_AD)+length(trVec_MCI)
        X = [X ; trVec_MCI(m-length(trVec_AD))];
        y = [y ; 0];
    end
    
    % combine two different populations in same test array
    testX = [];
    testy = [];
    for m=1:length(testVec_AD)
        testX = [testX ; testVec_AD(m)];
        testy = [testy ; 1];
    end
    for m=length(testVec_AD)+1:length(testVec_AD)+length(testVec_MCI)
        testX = [testX ; testVec_MCI(m-length(testVec_AD))];
        testy = [testy ; 0];
    end
    
    % train model
    SVMModel = fitcsvm(X,y);
    
    % test model
    [label,score] = predict(SVMModel,testX);
    modelstr{i} =SVMModel;
    labelstr{i} = label;
    scorestr{i} = score;
    actualLabelstr{i} = testy;
   

    perf = classperf(testy,label);
    perf.CorrectRate
    perf.Specificity
    perf.Sensitivity
end

travg_accu1=0;
travg_spec1=0;
travg_sens1=0;
for i=1:fold_k
    perf = classperf(actualLabelstr{i},labelstr{i});
    traccu{i} = perf.CorrectRate;
    trspec{i} = perf.Specificity;
    trsens{i} = perf.Sensitivity;
    travg_accu1 = travg_accu1 + perf.CorrectRate;
    travg_spec1 = travg_spec1 + perf.Specificity;
    travg_sens1 = travg_sens1 + perf.Sensitivity;
end
travg_accu1 = travg_accu1/fold_k;
travg_spec1 = travg_spec1/fold_k;
travg_sens1 = travg_sens1/fold_k;

ac = [travg_accu1 avg_accu];
sp = [travg_spec1 avg_spec];
se = [travg_sens1 avg_sens];

datasetNames{1}='original-train';
datasetNames{2}='genetic';

%%% accuracy
tiledlayout(2,2) % Requires R2019b or later
nexttile
b = bar(ac)
ylim([0 1]);
xlim([0 length(ac)+1]);
xlabel('Dataset');
ylabel('Accuracy');
title('Model Accuracy');
set(gca, 'XTick', 1:length(datasetNames),'XTickLabel',datasetNames);
xtickangle(0)
% first bar naming
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',45)

%%% specificity
nexttile
b = bar(sp)
ylim([0 1]);
xlim([0 length(sp)+1]);
xlabel('Dataset');
ylabel('specificity');
title('Model specificity');
set(gca, 'XTick', 1:length(datasetNames),'XTickLabel',datasetNames);
xtickangle(0)
% first bar naming
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',45)


%%% sensitivity
nexttile
b = bar(se)
ylim([0 1]);
xlim([0 length(se)+1]);
xlabel('Dataset');
ylabel('sensitivity');
title('Model sensitivity');
set(gca, 'XTick', 1:length(datasetNames),'XTickLabel',datasetNames);
xtickangle(0)
% first bar naming
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',45)