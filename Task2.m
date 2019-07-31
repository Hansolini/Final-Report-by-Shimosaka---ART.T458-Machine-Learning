set(0,'defaulttextInterpreter','latex');

%% 1. Proximal gradient of Lasso
% Parameters
A = [  3 0.5
     0.5   1];

mu = [1 2]';
lambdas = [2 4 6]';
L = max(eig(2*A));

w_0 = [3 -1]';

% Proximal gradient
num_epochs = 20;

figure(1);
clf;
semilogy(0,0,'HandleVisibility','off');
hold on; grid on;
title('Proximal Gradient of Lasso with varying $\lambda$');
xlabel('Iterations [t]');
ylabel('$||w^{(t)} - \hat{w}||$');
legend('interpreter','latex');
    
for lambda = lambdas'
    w_hist = w_0;
    loss_hist_prox = [];
    for i = 1:num_epochs
        grad = 2*A*(w_hist(:,i) - mu);
        w_th = w_hist(:, i) - 1/L * grad;
        w_hist(:, i+1) = prox(w_th, lambda*1/L);
    end

    % Solve w_hat with cvx
     cvx_begin
         variable w_hat(2,1)
         minimize((w_hat - mu)'*A*(w_hat - mu) + lambda*norm(w_hat, 1))
     cvx_end

    % Plot of the solution trajectory
    figure(1);
    semilogy(0:num_epochs, vecnorm(w_hist - w_hat), 'o-', 'DisplayName', ['$\lambda =$ ' num2str(lambda)]);
end

%% 2. Regularization path
lambdas = 0:0.1:10;

w_hat_lambda = [];

for i = 1:length(lambdas)
    w_hat = w_0;
    loss_hist_prox = [];
    for j = 1:num_epochs
        grad = 2*A*(w_hat - mu);
        w_hat_h = w_hat - 1/L * grad;
        w_hat = prox(w_hat_h, lambdas(i)*1/L);
    end
    w_hat_lambda(:, i) = w_hat;
end

figure(2);
clf;
plot(lambdas, w_hat_lambda, 'lineWidt', 2);
grid on;
title('Regularization path for lasso');
xlabel('$\lambda$');
ylabel('$w$');
legend('$w_{1}$','$w_{2}$','interpreter','latex');

% %% 3. Group lasso
% 
% % Dataset
% d = 200;
% n = 180;
% % we consider 5 groups where each group has 40 attributes
% g = cell(5, 1);
% for i = 1:length(g)
%     g{i} = (i-1)*40+1:i*40;
% end
% x = randn(n, d);
% noise = 0.5;
% % we consider feature in group 1 and group 2 is activated.
% w = [20 * randn(80, 1);
%     zeros(120, 1);
%     5 * rand];
% 
% x_tilde = [x, ones(n, 1)];
% y = x_tilde * w + noise * randn(n, 1);
% 
% lambda = 1.0;
% wridge = (x_tilde'*x_tilde + lambda * eye(d+1))\(x_tilde' * y);
% 
% % Solution with cvx
% cvx_begin
%     variable west(d+1,1)
%     minimize( 0.5 / n * (x_tilde * west - y)' * (x_tilde * west - y) + ...
%     lambda * ...
%    (norm(west(g{1}), 2.0) + ...
%     norm(west(g{2}), 2.0) + ...
%     norm(west(g{3}), 2.0) + ...
%     norm(west(g{4}), 2.0) + ...
%     norm(west(g{5}), 2.0) ))
% cvx_end
% 
% % Test dataset
% x_test = randn(n, d);
% x_test_tilde = [x_test, ones(n, 1)];
% y_test = x_test_tilde * w + noise * randn(n, 1);
% y_pred = x_test_tilde * west;
% mean((y_pred - y_test) .^2)
% 
% figure(3);
% clf;
% plot(west(1:d), 'r-o')
% 
% hold on
% plot(w, 'b-*');
% plot(wridge, 'g-');
% legend('group lasso', 'ground truth', 'ridge regression')
% figure(2);
% clf; 
% plot(y_test, y_pred, 'bs');
% xlabel('ground truth')
% ylabel('prediction')
% fprintf('carinality of w hat: %d\n', length(find(abs(west) < 0.01)))
% fprintf('carinality of w ground truth: %d\n', length(find(abs(w) < 0.01)))
%% Functions
% Objective funtion
%function J = objective_function(w, mu, A, lambda)
%    J = (w - mu)'*A*(w - mu) + lambda*norm(w, 1);
%end

% Proximal operator for lasso 
function p = prox(w, eta)
    p = zeros(length(w), 1);
    for i = 1:length(w)
        if w(i) > eta
            p(i) = w(i) - eta;
        elseif abs(w(i)) <= eta
            p(i) = 0;
        else
            p(i) = w(i) + eta;
        end
    end
end

