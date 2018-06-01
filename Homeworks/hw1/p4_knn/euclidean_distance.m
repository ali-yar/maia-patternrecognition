function dist = euclidean_distance (a, b)
  dist = 0;
  d = numel(a);
  
  for i = 1:d
    dist = dist + (a(i) - b(i))^2;
  end
  
  dist = sqrt(dist);
endfunction
