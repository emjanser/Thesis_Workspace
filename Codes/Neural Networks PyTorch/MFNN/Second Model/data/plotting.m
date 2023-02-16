close all
clear
clc

%% Importing the data
Z = table2array(readtable('Z.txt')).';
LF_Z = table2array(readtable('LF_Z.txt')).';
HF_Z = table2array(readtable('HF_Z.txt')).';

LF_PRED = table2array(readtable('LF_PRED.txt')).';
HF_PRED = table2array(readtable('HF_PRED.txt')).';
MF_PRED = table2array(readtable('MF_PRED.txt')).';

Y_LF_TRAIN = table2array(readtable('Y_LF_TRAIN.txt')).';
Y_HF_TRAIN = table2array(readtable('Y_HF_TRAIN.txt')).';

X_LF_TRAIN = table2array(readtable('X_LF_TRAIN.txt')).';
X_HF_TRAIN = table2array(readtable('X_HF_TRAIN.txt')).';

%% Plotting

% LF Plot
LF_FIGURE = figure('Renderer', 'painters', 'Position', [50 50 900 400]);
target = plot(Z, HF_Z, 'Color', [0, 0.275, 0.545], 'DisplayName', 'Target Sine Wave', 'LineWidth', 3);
hold on
LF_PLOT = plot(X_LF_TRAIN, Y_LF_TRAIN, 'o', 'MarkerFaceColor', 'r','DisplayName', 'LF Training Data Points', MarkerSize=8);
LF_POINTS = plot(Z, LF_PRED, "k", 'DisplayName', 'Low Fidelity Prediction', 'LineWidth', 2, "DisplayName", 'LOW Fidelity Prediction');
xlabel('X')
ylabel('Y')
grid("minor")
% title('LOW Fidelity Network Prediction','FontName',"Montserrat")
set(gca,'FontSize',11)
title(legend([target, LF_PLOT, LF_POINTS],'NumColumns',1,'location','southwest','FontSize',11),'Legend','FontSize',12);


% LH Plot
HF_FIGURE = figure('Renderer', 'painters', 'Position', [50 50 900 400]);
target = plot(Z, HF_Z, 'Color', [0, 0.275, 0.545], 'DisplayName', 'Target Sine Wave', 'LineWidth', 3);
hold on
LF_PLOT = plot(X_HF_TRAIN, Y_HF_TRAIN, 'o', 'MarkerFaceColor', 'b','DisplayName', 'LF Training Data Points', MarkerSize=8);
LF_POINTS = plot(Z, HF_PRED, "-k", 'DisplayName', 'Low Fidelity Prediction', 'LineWidth', 2, "DisplayName", 'HIGH Fidelity Prediction');
xlabel('X')
ylabel('Y')
grid("minor")
% title('HIGH Fidelity Network Prediction','FontName',"Montserrat")
set(gca,'FontSize',11)
title(legend([target, LF_PLOT, LF_POINTS],'NumColumns',1,'location','southwest','FontSize',11),'Legend','FontSize',12);


% LH Plot
MF_FIGURE = figure('Renderer', 'painters', 'Position', [50 50 900 400]);
target = plot(Z, HF_Z, 'Color', [0, 0.275, 0.545], 'DisplayName', 'Target Sine Wave', 'LineWidth', 3);
hold on
MF_POINTS = plot(Z, MF_PRED, "--k", 'DisplayName', 'Low Fidelity Prediction', 'LineWidth', 2, "DisplayName", 'MULTI-Fidelity Prediction');
xlabel('X')
ylabel('Y')
grid("minor")
% title('HIGH Fidelity Network Prediction','FontName',"Montserrat")
set(gca,'FontSize',11)
title(legend([target, MF_POINTS],'NumColumns',1,'location','southwest','FontSize',11),'Legend','FontSize',12);



%% Saving Figures
saveas(LF_FIGURE ,'LF_Figure.png')
saveas(HF_FIGURE ,'HF_Figure.png')
saveas(MF_FIGURE ,'MF_Figure.png')
