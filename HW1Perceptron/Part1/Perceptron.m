%% Training: Read Data
clear all;
close all;

fid = fopen('train.dat', 'r');
a= fscanf(fid, '%d',[51, 260]); 
inputs=a';

nclases = 26;               % the 26 letters of the alphabet
npixels = 25;               % the 25 pixels in each image
ninputs = size(inputs,1);   % size of the sample

xTrain=inputs(:,1:npixels);
xTrain = [ones(ninputs,1) xTrain]; %Add Extra input to account for Biases
yTrain=inputs(:,npixels+1:end);

%% Training: Plot first 26 Training Data points
figure
for r = 1:26
    img = xTrain(r,2:end);
    img = reshape(img,[5,5]); 
    subplot(6,5,r), subimage(img),
    set(gca,'visible','off')
end
set(gcf,'numbertitle','off','name','Training Data') 
suptitle('Sample Training Data')

%% Training: Initialize the weights and constants
weights = (randn(nclases,1+npixels)-randn(nclases,1+npixels))/nclases; % weights is matrix form because we need 11-training weghts
alpha =0.005;   % learning rate constant

%index variables:
epoch =1;       
sum_square_error = 1;
%% Training: perform training and Plot error
error = zeros(ninputs,nclases);
while sum_square_error > 0 % I used same contion here.
     %% Evaluate Neuron Output
     currentoutput = zeros(ninputs,nclases);
     for i=1:ninputs
        for k=1:nclases
            % calculate current network output for each class               
            if dot(xTrain(i,:),weights(k,:)) > 0
                currentoutput(i,k) = 1;
            else
                currentoutput(i,k) = 0;
            end
        end
     end    
     %% Calculate Error
     errorTrain = yTrain-currentoutput;
     error_square = errorTrain.^2;
     sum_square_error = sum(sum(error_square))/ninputs;
     sum_square_error_hist(epoch) = sum_square_error;   
     
     %% Update Weights using Generalized Delta Rule(batch training)
     for j=1:nclases
          for k = 1:npixels                                                              
               delta = alpha*dot(errorTrain(:,j),xTrain(:,k));
               weights(j,k) = weights(j,k) + delta;   
          end
     end
     epoch = epoch + 1;
end
%draw the graph of SSE versus Epochs  
figure                                     
plot(sum_square_error_hist);
hold off
title('SSE Change of the Perceptron Training')
xlabel('epochs')
ylabel('SSE')


%% Testing: Read Data
fid = fopen('test.dat', 'r');
T = fscanf(fid, '%d', [51, 260]);
tests=T';

xTest=inputs(:,1:npixels); 
xTest = [ones(ninputs,1) xTest];%Add Extra input to account for Biases

%add some noise by changing random bits in the training set
nbitsChanged = 100;
[m,n] = size(xTest);      
idx = randi([1 ninputs*npixels],1,nbitsChanged);
xTest(idx) = ~xTest(idx);

yTest=inputs(:,npixels+1:end);
testingoutput = zeros(ninputs,nclases);

%% Run the testing data through the network
for i=1:ninputs
  for k=1:nclases
        if dot(xTest(i,:),weights(k,:)) > 0
            testingoutput(i,k) = 1;
        else
            testingoutput(i,k) = 0;
        end
  end
end

%% Calculate Testing Error
errorTest = yTest-testingoutput;
error_squareTest = errorTest.^2;
sum_square_errorTest = double(sum(sum(error_squareTest)))/double(ninputs)

%% Display Weights graph
normA = weights - min(weights(:));
normA = normA ./ max(normA(:));
figure
imagesc(normA);
colorbar
title('LEARNED WEIGHTS MATRIX')
xlabel('Input pixel','FontSize',12,'FontWeight','bold')
ylabel('Output','FontSize',12,'FontWeight','bold')

