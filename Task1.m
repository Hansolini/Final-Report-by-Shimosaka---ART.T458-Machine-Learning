%% Task 1
set(0,'defaulttextInterpreter','latex');
seed = 17;
rng(seed);
 
%% Dataset IV
n = 200;
d = 4;
x = 3 * (rand(n, d) - 0.5);
y = (2 * x(:, 1) - 1 * x(:, 2) + 0.5 + 0.5 * randn(n, 1)) > 0;
y = 2 * y - 1;

lambda = 0.01;
 
%% Plot the data
figure(1);
clf;
hold on; grid on;
plot(x(y > 0, 1), x(y > 0, 2), 'ro', 'markersize', 16, 'linewidth', 3);
hold on;
plot(x(y < 0, 1), x(y < 0, 2), 'bx', 'markersize', 16, 'linewidth', 3);
 
xlabel('$x_1$', 'interpreter','latex')
ylabel('$x_2$', 'interpreter','latex')

axis image
axis([min(x(:,1)), max(x(:,1)), min(x(:,2)), max(x(:, 2))]);
drawnow;

%% 1. Batch Gradient descent
num_iter = 200;
w = [3;3;3;3];

ll_history = [];
w_history = [];
lip = 0.25 * max(max(x))^2;
lambda = 0.01;

% Iterations
tic
for t = 1:num_iter
    posteriori = 1.0 ./ (1.0 + exp(-y .* x*w ));
    grad =  1.0 / n * sum((-y).*x.*(1-posteriori))' + 2 * lambda * w;
    direction = -grad;
    
    ll = 1.0/n  * sum(log(1.0 + exp(-y.* x*w))) + lambda * w'*w;
    
    ll_history = [ll_history ll];
    w = w + 1.0 / sqrt(t) / lip * direction;

end
toc

% Plot
show_iter = 100;
figure(2);
clf;
hold on; grid on;
plot(1:show_iter, ll_history(1:show_iter), 'bo-');

title('Batch Gradient Descent method');
xlabel('Iterations [t]');
ylabel('$J(${\boldmath $w^{(t)}$}$)$');

 
 
%% 2. Newton method
ll_n_history = [];
ww_n_history = [];
w = [3;3;3;3];

% Iterations
tic
for t = 1:num_iter
    posteriori = 1.0 ./ (1.0 + exp(-y .* x*w ));
    grad =  1.0 / n * sum((-y).*x.*(1-posteriori))' + 2 * lambda * w;
    hess = 1.0 / n * ((posteriori .* (1.0 - posteriori) .*x)'*x) + 2 * lambda*eye(length(w));
    direction = - inv(hess)*grad;
    
    ll = 1.0/n  * sum(log(1.0 + exp(-y.* x*w))) + lambda * w'*w;
    
    ll_n_history = [ll_n_history ll];
    w = w + 1.0 / sqrt(t+10) * direction;
end
toc
show_iter = 100;

min_ll = min(min(ll_n_history), min(ll_history));

% Plot
show_iter = 100;
figure(3);
clf;
hold on; grid on;
plot(1:show_iter, ll_n_history(1:show_iter), 'ro-');

title('Newton method');
xlabel('Iterations [t]');
ylabel('$J(${\boldmath $w^{(t)}$}$)$');

%% 3. Comparison
show_iter = 100;
figure(4);
clf;
semilogy(1:show_iter,abs(ll_history(1:show_iter) - min_ll), 'bo-');
hold on; grid on;
semilogy(1:show_iter,abs(ll_n_history(1:show_iter) - min_ll), 'ro-');

title('Comparison of the Batch Gradient Descent method and the Newton method');
xlabel('Iterations [t]');
ylabel('$J(${\boldmath $w^{(t)}$}$) - J(${\boldmath$\hat{w}$}$)$');
legend('Batch Gradient Descent', 'Newton');
