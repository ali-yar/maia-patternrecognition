clear;
clc;

% pkg load statistics

function label = knnclassifier(Ts,x,k)
  len = size(Ts)(1);
  D = [];
  for i = 1:len
    D(i,:) = [norm(x-Ts(i,1:2)) Ts(i,3)];
  end
  D = sortrows(D);
  D = D(1:k,2);
  k1 = sum(D(:)==0);
  k2 = sum(D(:)==1);
  if k1>k2
    label = 0;
  else
    label = 1;
  end
end

P1 = 0.5; % probability P(w1)
N = 200; % total samples
N1 = N * P1; % total samples in class w1
N2 = N - N1; % total samples in class w2

mu1 = [2 3]; % mean/expected value of feature vector given w1
mu2 = [5 5]; % mean/expected value of feature vector given w2

sigma1 = [1    1.5;
         1.5    3]; % covariance matrix of w1
         
sigma2 = [2    1.5;
         1.5    5]; % covariance matrix of w2
         
sigma2 = sigma1;         

w1 = [mvnrnd(mu1,sigma1,N1) zeros(N1,1)]; % generate w1 samples with class label 0
w2 = [mvnrnd(mu2,sigma2,N2) ones(N2,1)]; % generate w2 samples with class label 1

trainRatio = 0.6;
trainSet = [w1(1:N1*trainRatio,:); w2(1:N2*trainRatio,:)];
testSet = [w1(N1*trainRatio+1:end,:); w2(N2*trainRatio+1:end,:)];

Nerror = 0;

len = size(testSet)(1);
for i = 1:len
  x = testSet(i,1:2); % sample to test
  y_truth = testSet(i,3); % ground truth
  y_found = knnclassifier(trainSet, x, 3); % classification

  if y_found != y_truth
    Nerror++;
  end
end

errorRate = Nerror / len;
accuracy = 1 - errorRate;

  

  