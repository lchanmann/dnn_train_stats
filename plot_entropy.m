% global constants
nH = '800';                                 % number of hidden units
colors = {'b-', 'g--', 'r-.', 'm:', 'c-'};  % colors for ploting
dnns = 3:7;                                 % dnn models

% variables
l = length(dnns);   % total models to be evaluated
M = cell(1, l);     % legend text
x = zeros(2, l);    % finetune points

% plot
figure; hold on

% Go through DNN models with different depth
for j=1:l
    i = num2str(dnns(j));
    % Load data
    [val_entropy, stage] ...
        = textread(['dnn' i '_' nH '.data'], '%*f,%*f,%*f,%f,%*f,%d');
    
    plot(val_entropy, colors{j});
    
    M{j} = ['DNN ' i];
    x(1, j) = find(stage == 2, 1);
    x(2, j) = val_entropy(x(1, j));
end
plot(x(1, :), x(2, :), 'kx');

% plot configurations
xlim([1 Inf]);
xlabel('# of Epoch');
ylabel('J/n');
title(['DNN with (' nH ' hidden units) validation cross entropy']);
legend(M);
