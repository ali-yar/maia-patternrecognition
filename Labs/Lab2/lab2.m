function accuracy = lab2(disciminantType = 1)

  data = load('pima-indians-diabetes.data'); % load data file

  classValues = data(:,9); % return last column containing class values

  w1 = data(find(classValues == 1),1:8); % extract rows with class value 1
  w2 = data(find(classValues == 0),1:8); % extract rows with class value 0

  len = length(data);
  len1 = length(w1); % number of sample of class 1
  len2 = length(w2); % number of sample of class 2

  p1 = len1 / len;
  p2 = len2 / len;

  trainSize = 0.5; 

  % train data set
  train1 = w1(1:floor(len1*trainSize), :);
  train2 = w2(1:floor(len2*trainSize), :); 

  % test data set               
  test1 = w1(floor(len1*trainSize)+1:end, :);
  test2 = w2(floor(len2*trainSize)+1:end, :);

  mu1 = mean(train1);
  mu2 = mean(train2);

  sigma1 = cov(train1);
  sigma2 = cov(train2);

  nError = 0; % error counter

    function res = discriminant(x, mu, sigma, p)
      if disciminantType == 1
        res = -0.5 * (x - mu) * inv(sigma) * (x - mu)' + log(p);
      else
        res = -0.5 * (x - mu) * inv(sigma) * (x - mu)' + log(p);
      end
    end
 sigma = sigma1* p1 + sigma2*p2;
    function runTest(test, classType)
      for i = 1:length(test)
        y1 = discriminant(test(i,:), mu1, sigma, p1);
        y2 = discriminant(test(i,:), mu2, sigma, p2);
        
        if (y1 > y2 && classType != 1) || (y1 < y2 && classType != 2)
           nError = nError + 1;
        end 
      end
    end

  runTest(test1, 1);
  runTest(test2, 2);
  
  errorRate = nError / (length(test1) + length(test2));
  
  accuracy = 1 - errorRate;
end