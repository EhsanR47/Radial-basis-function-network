%Ehsan Rassekh
close all
clear
clc

%read file
file = fopen('iris.data');

%extract data from the txt file using textscan function
text_data = textscan(file,'%f %f %f %f %s', 200, 'Delimiter',',');

%form the data matrix
data = cell2mat(text_data(:,1:4));
target=text_data{1,5};
[m,n] = size(target);
tr=[];

%form the target matrix
for k= 1:m
    a=target(k);
    if strcmp(a,'Iris-setosa')==1
        l=-1;
    elseif strcmp(a,'Iris-versicolor')==1
        l=0;
    else
        l=1;
    end
    tr=[tr;l];
end
clear a;

%merge both the matrix together

data=[data tr];

figure(1);
plot(data(:,3),data(:,4),'k*','MarkerSize',5)
title ' Iris Data';
xlabel 'Petal Lengths (cm)'; 
ylabel 'Petal Widths (cm)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%divide data to train and evaluate
% Cross validation (train: 80%, test: 20%)
cv = cvpartition(size(data,1),'HoldOut',0.2);
idx = cv.test;
% Separate to training and test data
dataTrain = data(~idx,:);
dataTest  = data(idx,:);


%this section implementation of kmeans 
k = 3;
[idx,C] = kmeans(dataTrain,k);
%display(size(C))
%display(size(data))
x1 = min(dataTrain(:,3)):0.01:max(dataTrain(:,3));
x2 = min(dataTrain(:,4)):0.01:max(dataTrain(:,4));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot

idx2Region = kmeans(XGrid,3,'MaxIter',100); %,'Start',C);

figure(2);
gscatter(XGrid(:,1),XGrid(:,2),idx2Region,...
    [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(dataTrain(:,3),dataTrain(:,4),'k*','MarkerSize',5);
title 'Iris Data';
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)'; 
legend('Region 1','Region 2','Region 3','Data','Location','SouthEast');
hold off;

%finding variance for calculate spread
m_max = 0;
for i = 1:k-1
    d_spread(i) = sum((C(i,:)-C(i+1,:)).^2);
    if  d_spread(i) > m_max
            m_max = d_spread(i);
    else
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
spread = (m_max/sqrt(2*k))*ones(k,1);

train_target = dataTrain(:,5);
dataTrain = dataTrain(:,1:4);
test_target = dataTest(:,5);
dataTest = dataTest(:,1:4);
C = C(:,1:4);
true = zeros(length(test_target),1);
for i = 1:length(test_target)
    if test_target(i) == 1
        true( i ) = 3;
    elseif test_target(i) == 0
        true( i ) = 2;
    else
        true( i) = 1;
    end
end

y_one_hot = zeros( size( train_target, 1 ), 3 );
for i = 1:length(train_target)
    if train_target(i) == 1
        y_one_hot( i, 3 ) = 1;
    elseif train_target(i) == 0
        y_one_hot( i, 2 ) = 1;
    else
        y_one_hot( i, 1 ) = 1;
    end
end

%%% finding weights
[m,n] = size(dataTrain);
params = rand(n,k);
goal = zeros(k,m);
for i=1:m
    temp = dataTrain(i,:);
    for j = 1:k
        goal(j,i) = exp(-(temp-C(j,:))*(temp-C(j,:))'/(2*spread(j)^2));
    end
end
params = pinv(goal)'* y_one_hot;

%%applying rbf to test set
for i = 1:length(dataTest)
    temp = dataTest(i,:);

    for j = 1:k
        goal_test(j,i) = exp(-(temp-C(j,:))*(temp-C(j,:))'/(2*spread(j)^2));
    end
end

%%%prediction and evaluate the model
prediction = (goal_test'* params);
for z = 1:length(dataTest)
    prediction(z) = softmax(prediction(z));
    [val, col] = max(prediction(z,:));
    predict(z) = col;
end
acc = (sum(predict==true')/length(true))*100;
display(acc)