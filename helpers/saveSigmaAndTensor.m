function [] = saveSigmaAndTensor(fileName,sigmasAD,sigmasMCI,rep_tensorsAD,rep_tensorsMCI)

save(fileName,'sigmasAD','sigmasMCI','rep_tensorsAD','rep_tensorsMCI');
