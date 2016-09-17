clc;
close all;
%Load the Iris_dataset implemented natively into matlab
[x,t] = iris_dataset;

%Create a feedforward NN that uses Gradient Descent
net = feedforwardnet(10,'trainlm');

%Do not open the train window
net.trainParam.showWindow = false;

%Train the neural network
[net,tr] = train(net,x,t);

% Plot Performance
plotperform(tr)


