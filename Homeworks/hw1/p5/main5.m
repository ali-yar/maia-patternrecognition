clear, clc;

load testdatafile.mat;
n = 1000;
testData = [testData(1:n,:); testData(8000-n+1:8000,:)];

A = testData(:,1:end-1);

% call the classifier function
y = test(A);

% compute accuracy
testGT = testData(:,end);
r = y .* testGT; % positive only if same sign
accuracy = length(find(r>0)) / numel(r);
