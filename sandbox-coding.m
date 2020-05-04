
rrA{5} = reptensorsAD{5};
rrM{5} = reptensorsMCI{5};
ssA{5} = sigmasAD{5};
ssM{5} = sigmasMCI{5};

%% getting the best sigma values deprecated
bestsigma_AD1 = best_sigmasAD{1};
bestsigma_MCI1 = best_sigmasMCI{1};
train_representative_tensor_AD1 = train_representative_tensor_AD;
train_representative_tensor_MCI1 = train_representative_tensor_MCI;
bestsigma_AD3 = best_sigmasAD{1};
bestsigma_MCI3 = best_sigmasMCI{1};
train_representative_tensor_AD3 = train_representative_tensor_AD;
train_representative_tensor_MCI3 = train_representative_tensor_MCI;



%%
sigmas_AD={};
sigmas_MCI={};
sigmas_AD{1}=bestsigma_AD1;
sigmas_AD{2}=bestsigma_AD2;
sigmas_AD{3}=bestsigma_AD3;
sigmas_MCI{1}=bestsigma_MCI1;
sigmas_MCI{2}=bestsigma_MCI2;
sigmas_MCI{3}=bestsigma_MCI3;

representative_tensor_AD = {};
representative_tensor_MCI = {};
representative_tensor_AD{1} = train_representative_tensor_AD1;
representative_tensor_AD{2} = train_representative_tensor_AD2;
representative_tensor_AD{3} = train_representative_tensor_AD3;
representative_tensor_MCI{1} = train_representative_tensor_MCI1;
representative_tensor_MCI{2} = train_representative_tensor_MCI2;
representative_tensor_MCI{3} = train_representative_tensor_MCI3;



sigmas_AD={};
sigmas_MCI={};
representative_tensor_AD = {};
representative_tensor_MCI = {};

sigmas_AD{5}=best_sigmasAD{1};
sigmas_MCI{5}=best_sigmasMCI{1};
representative_tensor_AD{5} = train_representative_tensor_AD;
representative_tensor_MCI{5} = train_representative_tensor_MCI;

%%
times=[];
tic
for i=1:500
    rng(i);
    sigma = rand(1,10);
    nearest_posdef(unVectorizeSigma(sigma));
end
times(1)=toc;

tic
for i=1:500
    rng(i);
    sigma = rand(1,10);
    nearestSPD(unVectorizeSigma(sigma));
end
times(2)=toc

% parelel computing
tic
parfor i=1:500
    rng(i);
    sigma = rand(1,10);
    nearest_posdef(unVectorizeSigma(sigma));
end
times(3)=toc;

tic
parfor i=1:500
    rng(i);
    sigma = rand(1,10);
    nearestSPD(unVectorizeSigma(sigma));
end
times(4)=toc

%%
sigmas_AD{1}=sigmasAD
sigmas_AD{2}=sigmasAD
sigmas_AD{3}=sigmasAD
sigmas_AD{4}=sigmasAD
sigmas_AD{5}=sigmasAD


sigmas_MCI{1}=sigmasMCI
sigmas_MCI{2}=sigmasMCI
sigmas_MCI{3}=sigmasMCI
sigmas_MCI{4}=sigmasMCI
sigmas_MCI{5}=sigmasMCI