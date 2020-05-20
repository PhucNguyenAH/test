function optional1
clear
data = readtable('ex1data1.txt'); % read tabular data into a table
data.Properties.VariableNames = {'Population','Profit'} % name the variables
summary(data)
methods(data)
linMdl = fitlm(data);
linMdl.Coefficients
theta = linMdl.Coefficients.Estimate;
fprintf('Theta computed by fitlm: [%f; %f]',theta(1),theta(2))

% Predict values for population sizes of 35,000 and 70,000
predict1 = predict(linMdl, 3.5);
fprintf('For population = 35,000, we predict a profit of %f', predict1*10000);
predict2 = predict(linMdl, 7);
fprintf('For population = 70,000, we predict a profit of %f', predict2*10000);
% Plot the linear fit
plot(data.Population, data.Profit, 'rx'); hold on
profit = predict(linMdl, data);
plot(data.Population, profit, '-')
legend('Training data', 'Linear regression'); hold off