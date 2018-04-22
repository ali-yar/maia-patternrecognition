pkg load statistics;
ratio = 0.5; % probability ratio: P(w2) / P(w1)
N1 = 200; % total samples in class w1
N2 = N1 * ratio; % total samples in class w2

P1 = N1 / (N1 + N2); % P(w1)
P2 = N2 / (N1 + N2); % P(w2)

mu1 = [2 3]; % mean/expected value of feature vector given w1
mu2 = [5 5]; % mean/expected value of feature vector given w2

sigma1 = [1    1.5;
         1.5    3]; % covariance matrix of w1
         
sigma2 = [1    1.5;
         1.5    3]; % covariance matrix of w2

r1 = mvnrnd(mu1,sigma1,N1); % generate w1 samples following normal distribution
r2 = mvnrnd(mu2,sigma2,N2); % generate w2 samples following normal distribution

figure;
plot(r1(:,1),r1(:,2),'+');
hold;
plot(r2(:,1),r2(:,2),'o');
x_axis = get(gca,'xLim'); y_axis = get(gca,'yLim');
x_vect = linspace(x_axis(1),x_axis(2),60);
y_vect = linspace(y_axis(1),y_axis(2),60);

Z1 = zeros(length(x_vect),length(y_vect));
Z2 = zeros(length(x_vect),length(y_vect));

for i=1:length(x_vect)
  for j=1:length(y_vect)
    Z1(i,j)= -0.5 * ([x_vect(i) y_vect(j)] - mu1) * inv(sigma1) * ([x_vect(i);y_vect(j)] - mu1') + log(P1);
    Z2(i,j)= -0.5 * ([x_vect(i) y_vect(j)] - mu2) * inv(sigma2) * ([x_vect(i);y_vect(j)] - mu2') + log(P2);
  end
end

[X Y] = meshgrid(x_vect, y_vect);
%imagesc(X(:),Y(:),(Z2-Z1)')
plot(r1(:,1),r1(:,2),'+');
plot(r2(:,1),r2(:,2),'o');
[C,h]=contour(X,Y,(Z2-Z1)',[0 0]);
%set(h,'ShowText','on','EdgeColor',[1 1 1]);


