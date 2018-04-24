function y = knn(trainSet,x,k)
  d = length(x); % dimensionality of the feature vector
  len = size(trainSet,1); % total samples to classify
  D = []; % euclidean distance vector
  for i = 1:len
    D(i,:) = [norm(x-trainSet(i,1:d)) trainSet(i,end)]; % compute Eucl. distance
  end
  D = sortrows(D); % sort the array by increasing distances
  D = D(1:k,2); % keep only the labels of the k lowest distances
  k1 = sum(D(:)==1); % total samples belonging to class P
  k2 = sum(D(:)==-1); % total samples belonging to class N
  y = (k1 - k2) / k;
end