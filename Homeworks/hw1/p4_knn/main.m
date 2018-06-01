clear, clc;

k = 15; % k nearest neighbors
repeat = 1; % number of times to repeat the classification

avgTPR = 0;
avgAcc = 0;
avgAUC = 0;

for i=1:repeat
  ratio = 0.75; % ratio from each class to build the training set
  
  % load data from file
  load hw1data.mat;
  data = Bdata;

  % for rapid testing only
  % data = [data(find(data(:,end)==1),:)(1:1000,:); data(find(data(:,end)==-1),:)(1:1000,:)];

  d = size(data,2) - 1; % dimension of feature vector

  % normalization params
  meanData = mean(data(:,1:d));
  stdData = std(data(:,1:d));

  % normalize data
  for i = 1:size(data,1);
    data(i,1:d) = (data(i,1:d) - meanData) ./ stdData;
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
  
  if i == 1
    RChPlot(R,Q,"KNN (k=50)");
  end
  
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

  TPRatFPR = R(find(R(:,1)-0.1>0,1)-1,2); % TPR corresponding to FPR = 0.1

  % compute accuracy
  r = y .* testGT'; % positive only if same sign
  accuracy = length(find(r>0)) / len;
  
  avgTPR  = avgTPR + TPRatFPR;
  avgAcc = avgAcc + accuracy;
  avgAUC  = avgAUC + a;
 end
 
 avgTPR = avgTPR / repeat
 avgAcc = avgAcc / repeat
 avgAUC = avgAUC / repeat