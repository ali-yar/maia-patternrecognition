type = 'Linear'; % 'linear' for linear, else for quadratic

ratio = 0.75; % ratio from each class to build the training set

% load data from file
load hw1data.mat;
data = Bdata;

[rows cols]= size(data); % get # of rows and columns

d = cols - 1; % dimension of feature vector

samplesP = data(find(data(:,end)==1),:); % get class P (labeled as 1) 
samplesN = data(find(data(:,end)==-1),:); % get class N (labeled as -1)

totalSamplesP = size(samplesP,1);
totalSamplesN = size(samplesN,1);

priorP = totalSamplesP / (totalSamplesP + totalSamplesN); % probability class P
priorN = totalSamplesN / (totalSamplesP + totalSamplesN); % probability class N

samplesP = samplesP(randperm(totalSamplesP),:); % randomly shuffle samples P
samplesN = samplesN(randperm(totalSamplesN),:); % randomly shuffle samples N

trainSetP = samplesP(1:totalSamplesP*ratio,1:d);
trainSetN = samplesN(1:totalSamplesN*ratio,1:d);

trainSet = [ trainSetP ; trainSetN ]; % build training set
             
testSet = [ samplesP(totalSamplesP*ratio+1:end,:) ;
            samplesN(totalSamplesN*ratio+1:end,:) ]; % build testing set

trainGT = [ samplesP(1:totalSamplesP*ratio,end) ; samplesN(1:totalSamplesN*ratio,end) ]; % ground truth for training set  
testGT = testSet(:,end); % ground truth for testing set (+1 or -1)

muP = mean(trainSetP); % the mean feature vector in class P
muN = mean(trainSetN); % the mean feature vector in class N

sigmaP = cov(trainSetP); % covariance matrix for class P
sigmaN = cov(trainSetN); % covariance matrix for class N

% if linear classifier, compute combined covariance
if strcmp(lower(type),'linear')
  sigma = sigmaP * priorP + sigmaN * priorN; % covariance for linear classifier
  sigmaP = sigma;
  sigmaN = sigma;
end