a = -10;
b = 15;
ts = 1e-4;
x = a:ts:b;

% prior probabilities
p_w1 = 1/2;
p_w2 = 1/2;

mu1 = 0;
mu2 = 7;

sigma1 = sqrt(15/2);
sigma2 = sqrt(7);

k1 = 1/(sqrt(2*pi)*sigma1);
k2 = 1/(sqrt(2*pi)*sigma2);

% conditional probabilities
p_x_w1 = k1 * exp( (-(x-mu1).^2) / (2*sigma1^2) );
p_x_w2 = k2 * exp( (-(x-mu2).^2) / (2*sigma2^2) );

% a posteori probabilities
p_w1_x = p_w1 * p_x_w1;
p_w2_x = p_w2 * p_x_w2;

g = p_w1_x - p_w2_x;

j = a;
for i=1:length(g)
  if g(i) < 0
    j = i-1;
    break;
  end
end

plot(x,p_x_w1);
hold;
plot(x,p_x_w2);
hold;
line(a+j*ts,0:1);
title("R1                                                     R2");
xlabel("x");
ylabel("p(x|wi)");
legend("p(x|w1)","p(x|w2)");
