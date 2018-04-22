clear, clc;

k = 50; % k nearest neighbors

ratio = 0.75; % ratio from each class to build the training set

data = load('hw1data.mat').Bdata; % load data from file

% for rapid testing only
%data = [data(find(data(:,end)==1),:)(1:100,:); data(find(data(:,end)==-1),:)(1:100,:)];

d = size(data,2) - 1; % dimension of feature vector

% normalization params
meanData = mean(data(1:d));
stdData = std(data(1:d));

% normalize data
for i = 1:size(data,1);
  data(i,1:d) = (data(i,1:d) - meanData) / stdData;
end

samplesP = data(find(data(:,end)==1),:); % get class P (labeled as 1) 
samplesN = data(find(data(:,end)==-1),:); % get class N (labeled as -1)

totalSamplesP = size(samplesP,1);
totalSamplesN = size(samplesN,1);

samplesP = samplesP(randperm(totalSamplesP),:); % randomly shuffle samples P
samplesN = samplesN(randperm(totalSamplesN),:); % randomly shuffle samples N

% build training set
trainSet = [ samplesP(1:totalSamplesP*ratio,:) ; 
             samplesN(1:totalSamplesN*ratio,:) ];

% build testing set
testSet = [ samplesP(totalSamplesP*ratio+1:end,:) ;
            samplesN(totalSamplesN*ratio+1:end,:) ];
            
testGT = testSet(:,end); % ground truth for testing set (+1 or -1)

y = [];

len = size(testSet,1);

for i = 1:len
  x = testSet(i,1:d); % sample to test
  y = [y knn(trainSet, x, k)];
end

[R,a] = EvalROC([testGT y']); % find ROC points (FPR,TPR) and AUC
Q = QHull(R);
RChPlot(R,Q,"KNN (k=50");

TPR = 0; % True Positive Rate
FPR = 0; % False Positive Rate

for i=1:len
  if y(i)==1 && testGT(i) == 1
    TPR = TPR + 1;
  elseif y(i)==1 && testGT(i) == -1
    FPR = FPR + 1;
  end
end

totalTestP = sum(testGT==1); % total samples P in testing set
totalTestN = sum(testGT==-1); % total samples N in testing set

TPR = TPR / totalTestP;
FPR = FPR / totalTestN;

% compute accuracy
r = y .* testGT'; % positive only if same sign
accuracy = length(find(r>0)) / len;