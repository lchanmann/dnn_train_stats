% global constants
nH = 800:200:2000;                          % number of hidden units
colors = {'b-', 'g--', 'r-.', 'm:', 'c-'};  % colors for ploting
dnns = 3:7;                                 % dnn models

% variables
l = length(dnns);   % total models to be evaluated

for i=1:length(nH)
    M = cell(1, l);         % legend text
    x = zeros(2, l);        % finetune points
    nodes = num2str(nH(i)); % number of hidden nodes

    % plot
    figure; hold on

    % Go through DNN models with different depth
    for j=1:l
        k = num2str(dnns(j));
        % Load data
        [val_entropy, stage] ...
            = textread(['dnn' k '_' nodes '.data'], '%*f,%*f,%*f,%f,%*f,%d');

        plot(val_entropy, colors{j});

        M{j} = ['DNN ' k];
        x(1, j) = find(stage == 2, 1);
        x(2, j) = val_entropy(x(1, j));
    end
    plot(x(1, :), x(2, :), 'kx');

    % plot configurations
    xlim([1 Inf]);
    xlabel('# of Epoch');
    ylabel('J/n');
    title(['DNN with (' nodes ' hidden units) validation cross entropy']);
    legend(M);
end
