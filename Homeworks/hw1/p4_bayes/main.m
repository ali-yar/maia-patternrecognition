clear, clc;

avgTPR = 0;
avgAcc = 0;

N = 3; % number of times to repeat the classification
for i=1:N
  init;

  gP = [];
  gN = [];
  len = size(testSet,1);
  for i = 1:len
    x = testSet(i,1:end-1); % feature vector
    % for a feature vector, compute the discriminant function with every class
    g = BayesDiscriminator(x,[muP;muN],[sigmaP;sigmaN],[priorP priorN],d,type);
    gP = [gP g(1)];
    gN = [gN g(2)];
  end

  g = gP - gN; % discriminants difference

  % compute accuracy
  y = g .* testGT'; % negative(found) * negative(ground truth) => positive(true)
  accuracy = length(find(y>0)) / size(testSet,1) ;
  avgAcc = avgAcc + accuracy;

  [R,a] = EvalROC([testGT g']); % find ROC points (FPR,TPR) and AUC
  Q = QHull(R);
  %RChPlot(R,Q,type + " Classifier");
  
  TPR = 0; % True Positive Rate
  FPR = 0; % False Positive Rate

  for i=1:len
    if g(i) > 0 && testGT(i) == 1
      TPR = TPR + 1;
    elseif g(i) > 0 && testGT(i) == -1
      FPR = FPR + 1;
    end
  end

  totalTestP = sum(testGT==1); % total samples P in testing set
  totalTestN = sum(testGT==-1); % total samples N in testing set

  TPR = TPR / totalTestP;
  FPR = FPR / totalTestN;

  TPRatFPR = R(find(R(:,1)==0.1)(end),2); % TPR corresponding to FPR = 0.1
  avgTPR = avgTPR + TPRatFPR; 
end

avgTPR = avgTPR / N;
avgAcc = avgAcc / N;

RChPlot(R,Q,strcat(type, " Classifier"));