% global constants
l = '7';    % total number of layers
nH = '800'; % number of hidden units

% Load data
[train_accuracy, val_accuracy, train_entropy, val_entropy, learning_rate, stage] ...
    = textread(['dnn' l '_' nH '.data'], '%f,%f,%f,%f,%f,%d');

% finetuning epoch
x = find(stage == 2, 1);

% plot accuracy
figure;
plot(train_accuracy); hold on
plot(val_accuracy, 'r--');
plot(x, train_accuracy(x), 'k*');
plot(x, val_accuracy(x), 'k*'); hold off
xlabel('# of Epoch');
ylabel('Accuracy');
title(['DNN ' l ' (' nH ' hidden units) training accuracy']);
legend('Train', 'Validation', 'Location', 'Best');

% plot cross entropy
figure;
plot(train_entropy); hold on
plot(val_entropy, 'r--');
plot(x, train_entropy(x), 'k*');
plot(x, val_entropy(x), 'k*'); hold off
xlabel('# of Epoch');
ylabel('J/n');
title(['DNN ' l ' (' nH ' hidden units) training cross entropy']);
legend('Train', 'Validation', 'Location', 'Best');

% plot learning rate
figure;
plot(learning_rate); hold on
plot(x, learning_rate(x), 'k*'); hold off
xlabel('# of Epoch');
ylabel('Learning rate');
title(['DNN ' l ' (' nH ' hidden units) learning rate']);
