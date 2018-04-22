function [Q,a]=QHull(A)
% [Q,a]=qhull(A)
%
% Calculates the coordinates of the vertices of the convex hull of the ROC
% curve, whose points are provided in the matrix A (Mx3; M points of the ROC curve: 
% for each point, the coordinates FPR,TPR are in the first two columns; the
% third column contains the threshoold value).
% 
% The convex hull is calculated by means of the algorithm QuickHull 
% ideated by C.B. Barber and H. Huhdanpaa, The Geometry Center,
% University of Minnesota. 

% Calls the Matlab function for calculating the convex hull
t = convhulln(A(:,1:2),{'Qt','Pp'});

% Sort the indices 
ind = sort(t(:,1));

% Build the matrix containing the coordinates of the convex hull, extracted
% from the matrix A. Each point contains the coordinates (FPR,TPR) and the
% threshold value
Q=A(ind,:);

% Calcola l'area del convex hull
a=0;
for i=2:length(Q)
    a=a+(Q(i,2)+Q(i-1,2))*(Q(i,1)-Q(i-1,1))/2;
end

% Check for the presence of erroneous points

% if (a > 0.5)
    erase=1;
    while(erase==1)
        erase=0;
        G=slopeval(Q);
        v=[];
        for i=2:length(G)-1
            if(G(i)<G(i-1) & G(i)<G(i+1))
                v=[v;i];
                erase=1;
            end
        end
        Q(v,:)=[];
    end
% end


% watchg(Q) 
% display(Q)

function G=slopeval(Q)
[U,Col]=size(Q);
G(1)=Inf;
for i=2:U
    den=Q(i,1)-Q(i-1,1);
    if(den==0)
        G(i)=Inf;
    else
        G(i)=(Q(i,2)-Q(i-1,2))/den;
    end    
end
G(U+1)=0;



function watchg(Q)

[U,Col]=size(Q);
G(1)=Inf;
for i=2:U
    den=Q(i,1)-Q(i-1,1);
    if(den==0)
        G(i)=Inf;
    else
        G(i)=(Q(i,2)-Q(i-1,2))/den;
    end    
end
G(U+1)=0;
G=G';
display(G)
