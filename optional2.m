clear;
% Load Data
data = readtable('ex1data2.txt');
data.Properties.VariableNames = {'Size', 'Bedrooms', 'Price'}
% linMdl = fitlm(data);
% theta = linMdl.Coefficients.Estimate;
% fprintf('Theta computed by fitlm: [%f; %f]', theta(1), theta(2))
% price = predict(linMdl, [1650 3]);
% fprintf('Predicted price of a 1650 sq-ft, 3 br house: $%f', price);

