r1 = % Samples of class 1
r2 = % Samples of class 1
figure
plot(r1(:,1),r1(:,2),'+');
hold
plot(r2(:,1),r2(:,2),'o');
x_axis = get(gca,'xLim'); y_axis = get(gca,'yLim');
x_vect = linspace(x_axis(1),x_axis(2),60);
y_vect = linspace(y_axis(1),y_axis(2),60);
Z1 = zeros(length(x_vect),length(y_vect));
Z2 = zeros(length(x_vect),length(y_vect));
for i=1:length(x_vect)
for j=1:length(y_vect)
Z1(i,j)= % value of the DF1 on [x_vect(i) y_vect(j)]'
Z2(i,j)= % value of the DF2 on [x_vect(i) y_vect(j)]'
end
end
[X Y] = meshgrid(x_vect, y_vect);
imagesc(X(:),Y(:),(Z2-Z1)')
plot(r1(:,1),r1(:,2),'+');
plot(r2(:,1),r2(:,2),'o');
[C,h]=contour(X,Y,(Z2-Z1)',[0 0]);
set(h,'ShowText','on','EdgeColor',[1 1 1]);
