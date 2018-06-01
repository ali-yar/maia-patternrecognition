function g = BayesDiscriminator (x, mu, sigma, prior)
  g = [];
  dim = length(x);
  n = length(prior);
  for i=1:n
    iStart = (i-1)*dim+1;
    sigmai = sigma(iStart:iStart+dim-1, :);
    a = - 0.5 * (x - mu(i,:)) * inv(sigmai) * (x - mu(i,:))';
    b = - 0.5 * log(det(sigma(i)));
    c = - (dim / 2) * log(2*pi);
    d = log(prior(i));
    g(i) = a + b + c + d;
  end
end
