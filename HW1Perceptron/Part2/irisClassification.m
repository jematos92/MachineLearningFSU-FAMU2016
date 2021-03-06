clc;
close all;
clear all;

%Load the Iris_dataset implemented natively into matlab
[x,t] = iris_dataset;

%% Plot the Data set:
targets = zeros(size(t,2),1);
for i = 1:size(t,2)
    targets(i)=find(t(:,i));
end

figure;
subplot(2,3,1), gscatter(x(1,:),x(2,:),targets), xlabel('Sepal length in cm'); ylabel('Sepal width in cm');
subplot(2,3,2), gscatter(x(1,:),x(3,:),targets), xlabel('Sepal length in cm'); ylabel('Petal length in cm');
subplot(2,3,3), gscatter(x(1,:),x(4,:),targets), xlabel('Sepal length in cm'); ylabel('Petal width in cm');
subplot(2,3,4), gscatter(x(2,:),x(3,:),targets), xlabel('Sepal width in cm'); ylabel('Petal length in cm');
subplot(2,3,5), gscatter(x(2,:),x(4,:),targets), xlabel('Sepal width in cm'); ylabel('housing');
subplot(2,3,6), gscatter(x(3,:),x(4,:),targets), xlabel('Petal length in cm'); ylabel('Petal width in cm');
suptitle('The Iris Dataset')

%% Netwok Training
%Create a feedforward NN that uses Gradient Descent
net = feedforwardnet();
%Train the neural network
[net,tr] = train(net,x,t);
% Plot Performance
figure
plotperform(tr)





