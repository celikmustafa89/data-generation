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

%% create dataset
% 35 AD samples
% 35 MCI samples
X_AD_tensors = {};
for i=1:35 % get 35 AD data samples
    X_AD_tensors{i}=trainTensor{i};
end
X_MCI_tensors = {};
for i=43:77 % get 35 MCI data samples
    X_MCI_tensors{i-42}=trainTensor{i};
end



%% create folds for cross-validation
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

%%
% seperate test samples
datasetTest_AD = X_AD_tensor_folds{3};
datasetTest_MCI = X_MCI_tensor_folds{3};

% separate train samples
t_index = 1;
datasetTrain_AD={};
datasetTrain_MCI={};
for j=1:fold_k
    if j ~= 3
        for k=1:fold_size
            for v=1:nv
                datasetTrain_AD{v}(:,:,t_index)  = X_AD_tensor_folds{j}{k}(:,:,v);
                datasetTrain_MCI{v}(:,:,t_index) = X_MCI_tensor_folds{j}{k}(:,:,v);
            end
            t_index = t_index + 1;
        end
    end
end

%% genetic algorithm and newViews of generated samples
% we are finding best sigma values for AD and MCI populations seperately.
% we will use these sigma values for creating fake datasets.
trainN=size(datasetTrain_AD{1},3);
[LH_train_NumFeatures_AD,LH_train_Frob_dist_AD,train_representative_tensor_AD,LH_train_netNorm_CBT_AD] = netNorm_func(datasetTrain_AD,nv,trainN,ROIs);
[best_sigmasAD, sigma_distancesAD, newViews_AD] = geneticAlgorithm(datasetTrain_AD,numIteration,trainN,nv,ROIs,train_representative_tensor_AD,LH_train_NumFeatures_AD,'half','version1','gen1');

trainN=size(datasetTrain_MCI{1},3);
[LH_train_NumFeatures_MCI,LH_train_Frob_dist_MCI,train_representative_tensor_MCI,LH_train_netNorm_CBT_MCI] = netNorm_func(datasetTrain_MCI,nv,trainN,ROIs);
[best_sigmasMCI, sigma_distancesMCI, newViews_MCI] = geneticAlgorithm(datasetTrain_MCI,numIteration,trainN,nv,ROIs,train_representative_tensor_MCI,LH_train_NumFeatures_MCI,'half','version1','gen1');


%% vectorize features for test samples
% test samples will be same for all datasets.
% so, we are doing it once, and at the beginning.
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

datasetNames{1}='original-train';
%% generate new datasets and models
datasetItr =1;
for i=1:datasetItr
    main_name='genetic(N+';
    dataset_name='*N)';
    datasetNames{i+1}=append(main_name,num2str(i),dataset_name);
    numOfNewSamples = 28*i;
    numOfOriginalSamples = 28;
    NumFeatures=595;
    [~, newViews_AD]  = generateSampleGA(train_representative_tensor_AD ,numOfNewSamples,nv,NumFeatures,ROIs, best_sigmasAD{1});
    [~, newViews_MCI] = generateSampleGA(train_representative_tensor_MCI,numOfNewSamples,nv,NumFeatures,ROIs, best_sigmasMCI{1});

    % vectorize features from generated samples
    trainVec_AD = [];
    for t=1:numOfNewSamples
        features = [];
        for u=1:nv
            features = [features vectorize(newViews_AD{u}(:,:,t))];
        end
        trainVec_AD = [trainVec_AD;features];
    end
    trainVec_MCI = [];
    for t=1:numOfNewSamples
        features = [];
        for u=1:nv
            features = [features vectorize(newViews_MCI{u}(:,:,t))];
        end
        trainVec_MCI = [trainVec_MCI;features];
    end
    % vectorize features from original samples
    %[~,trN]=size(datasetTrain_AD);
    for t=1:numOfOriginalSamples
        features = [];
        for u=1:nv
            features = [features vectorize(datasetTrain_AD{u}(:,:,t))];
        end
        trainVec_AD = [trainVec_AD;features];
    end
    %[~,trN]=size(datasetTrain_MCI);
    for t=1:numOfOriginalSamples
        features = [];
        for u=1:nv
            features = [features vectorize(datasetTrain_MCI{u}(:,:,t))];
        end
        trainVec_MCI = [trainVec_MCI;features];
    end
    % combine two different populations(AD and MCI) in same train array
    X = [];
    y = [];
    for m=1:size(trainVec_AD,1)
        X = [X ; trainVec_AD(m)];
        y = [y ; 1];
    end
    for m=size(trainVec_AD,1)+1:size(trainVec_AD,1)+size(trainVec_MCI,1)
        X = [X ; trainVec_MCI(m-size(trainVec_AD,1))];
        y = [y ; 0];
    end


    % train model
    SVMModel = fitcsvm(X,y);

    % test model
    [label,score] = predict(SVMModel,testX);
    models{i} =SVMModel;
    labels{i} = label;
    scores{i} = score;
    actualLabels{i} = testy;
    perf = classperf(testy,label);
    accu{i} = perf.CorrectRate;
    spec{i} = perf.Specificity;
    sens{i} = perf.Sensitivity;
end 


    ind=5;
    TR_datasetTest_AD = X_AD_tensor_folds{1};
    TR_datasetTest_MCI = X_MCI_tensor_folds{1};

    % vectorize features for test samples
    [~,TR_testN]=size(TR_datasetTest_AD);
    TR_testVec_AD = [];
    for t=1:TR_testN
        features = [];
        for u=1:nv
            features = [features vectorize(TR_datasetTest_AD{t}(:,:,u))];
        end
        TR_testVec_AD = [TR_testVec_AD;features];
    end
    [~,TR_testN]=size(TR_datasetTest_MCI);
    TR_testVec_MCI = [];
    for t=1:TR_testN
        features = [];
        for u=1:nv
            features = [features vectorize(TR_datasetTest_MCI{t}(:,:,u))];
        end
        TR_testVec_MCI = [TR_testVec_MCI;features];
    end
    % combine two different populations in same test array
    TR_testX = [];
    TR_testy = [];
    for m=1:length(TR_testVec_AD)
        TR_testX = [TR_testX ; TR_testVec_AD(m)];
        TR_testy = [TR_testy ; 1];
    end
    for m=length(TR_testVec_AD)+1:length(TR_testVec_AD)+length(TR_testVec_MCI)
        TR_testX = [TR_testX ; TR_testVec_MCI(m-length(TR_testVec_AD))];
        TR_testy = [TR_testy ; 0];
    end


    %%%%%%%%%%%%   ORIGINAL DATA SAMPLES N-FOLD TRAIN & TEST %%%%%%%%%%%%%
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

    end
    
%% % get the average result of the original model
TR_avg_accu1=0;
TR_avg_spec1=0;
TR_avg_sens1=0;
for i=1:fold_k
    perf = classperf(actualLabelstr{i},labelstr{i});
    TR_accu{i} = perf.CorrectRate;
    TR_spec{i} = perf.Specificity;
    TR_sens{i} = perf.Sensitivity;
    TR_avg_accu1 = TR_avg_accu1 + perf.CorrectRate;
    TR_avg_spec1 = TR_avg_spec1 + perf.Specificity;
    TR_avg_sens1 = TR_avg_sens1 + perf.Sensitivity;
end
TR_avg_accu1 = TR_avg_accu1/fold_k;
TR_avg_spec1 = TR_avg_spec1/fold_k;
TR_avg_sens1 = TR_avg_sens1/fold_k;


% combine all original train and newly generated model results
ac = [TR_avg_accu1];
sp = [TR_avg_spec1];
se = [TR_avg_sens1];
for i=1:datasetItr
    ac = [ac accu{i}];
    sp = [sp spec{i}];
    se = [se sens{i}];
end


%% graphs    

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
xtickangle(45)
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
xtickangle(45)
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
xtickangle(45)
% first bar naming
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',45)

%%% balanced accuracy
nexttile
b = bar((se+sp)/2)
ylim([0 1]);
xlim([0 length(se)+1]);
xlabel('Dataset');
ylabel('balanced accuracy');
title('Model Balanced Accuracy');
set(gca, 'XTick', 1:length(datasetNames),'XTickLabel',datasetNames);
xtickangle(45)
% first bar naming
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',45)
    