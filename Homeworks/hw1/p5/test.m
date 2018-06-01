%
% input   : testSet --> matrix (n x d) of samples to classify
% output  : y       --> vector (n x 1  ) of predicted class of samples (1 or -1)
%
function y = test( testSet )
    k = 29; % k nearest neighbors
    ratio = 0.85; % ratio to build train set from whole data
    d = size(testSet,2); % dimension of feature vector

    % load train data
    load data1.mat data;

    samplesP = data(data(:,end)==1,:);  % get class P (labeled as +1)
    samplesN = data(data(:,end)==-1,:); % get class N (labeled as -1)

    totalSamplesP = size(samplesP,1); % total samples in class P
    totalSamplesN = size(samplesN,1); % total samples in class N

    % build training set according to ratio
    trainSet = [ samplesP(1:totalSamplesP*ratio,:);
                 samplesN(1:totalSamplesN*ratio,:) ];

    % normalization params
    meanData = mean(trainSet(:,1:d));
    stdData = std(trainSet(:,1:d));
    
    lenTrain = size(trainSet,1);
    lenTest = size(testSet,1);
    
    % normalize train data
    for i = 1:lenTrain
        trainSet(i,1:d) = (trainSet(i,1:d) - meanData) ./ stdData;
    end

    % normalize test data
    for i = 1:lenTest
        testSet(i,1:d) = (testSet(i,1:d) - meanData) ./ stdData;
    end

    y = zeros(lenTest,1); % output initialized
    for j = 1:lenTest
        x = testSet(j,1:d); % sample to test
        D = zeros(lenTrain,2);
        for i = 1:lenTrain
            D(i,:) = [norm(x-trainSet(i,1:d)) trainSet(i,end)]; % compute e. distance
        end
        D = sortrows(D); % sort the array by increasing distances
        D = D(1:k,2); % keep only the labels of the k lowest distances
        k1 = sum(D(:)==1); % total samples belonging to class P
        k2 = sum(D(:)==-1); % total samples belonging to class N
        decision = 1;
        if k1 < k2
            decision = -1;
        end
        y(j) = decision;
    end

end