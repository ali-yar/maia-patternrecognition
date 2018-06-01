clear, clc;

a = -1;
b = 3;
ts = 1e-4;
x = a:ts:b;

len = length(x);

mu1 = 0;
mu2 = 1.5;

sigma1 = 0.1;
sigma2 = 0.3;

k1 = 1/(sqrt(2*pi)*sigma1);
k2 = 1/(sqrt(2*pi)*sigma2);

pw1 = 9999/10000;
pw2 = 1/10000;

p_x_w1 = k1 * exp( - (x-mu1).^2 / (2*sigma1^2) );
p_x_w2 = k2 * exp( - (x-mu2).^2 / (2*sigma2^2) );

plot(x, p_x_w1);
hold;
plot(x, p_x_w2);
hold;
line(-0.825,0:4);
line(0.45,0:4);
title("Healthy                                                  Sick");
xlabel("x");
ylabel("p(x|wi)");
legend("p(x|w1)","p(x|w2)");


