%% Task 3
set(0,'defaulttextInterpreter','latex');
seed = 11;
rng(seed);

%% Dataset II
n = 40;
omega = randn(1, 1);
noise = 0.8 * randn(n, 1);
x = randn(n, 2);
y = 2 * (omega * x(:, 1) + x(:, 2) + noise > 0) - 1;

figure(1)
clf;
hold on; grid on;
plot(x(:,1), x(:,2), 'r.', 'markerSize', 20)
title('Dataset II')
xlabel('$x_1$')
ylabel('$x_2$')

%% 3. Implementation of PGD
%% Plotting the objective function
lambda = 0.1;
w1 = -10:0.1:10;
w2 = w1;

o = [];

for i = 1:length(w1)
    for j = 1:length(w2)
        W = [w1(i);w2(j)];
        o(i, j) = primal_objective(y, W, x, lambda);
    end
end


[i, j] = find(o == min(min(o)));

% Plotting the objective funtion
figure(2);
clf;
surf(w1, w2, o');
hold on; grid on;

plot3(w1(i), w2(j), o(i,j),'b.', 'markersize',20);
colormap summer
shading interp

title('Hinge loss with regularization objective function')
xlabel('$w_1$')
ylabel('$w_2$')
zlabel('Hinge loss with regularization value')

%% Compute K for the dual problem
K = [];
for i = 1:length(y)
    for j = 1:length(y)
       K(i,j) = y(i)*y(j)*x(i,:)*x(j,:)'; 
    end
end

%% Optimization using CVX
lambda = 0.1;

% Optimization of the primal problem
cvx_begin
    variable w(2,1)
    minimize(sum(max(0, 1 - y.*x*w)) + lambda*power(2,norm(w)))
cvx_end

% Optimization of the dual problem
cvx_begin
    variable alpha_hat(n,1)
    maximise(-(1/(4*lambda))*alpha_hat'*K*alpha_hat + alpha_hat'*ones(n,1))
    subject to
        0 <= alpha_hat <= 1
cvx_end

disp('Computed w_hat from alpha')
disp(compute_w(alpha_hat, y, x, lambda))
disp('Solver w_hat')
disp(w)
disp('Computed optimal objective')
disp(primal_objective(y, w, x, lambda))
disp('Computed optimal dual-objective')
disp(dual_objective(alpha_hat, K, lambda))
disp('Actual optimal value')
disp(cvx_optval)

%% Projected Gradient Descent
alpha_0 = rand([n 1]);
alpha = alpha_hat.*alpha_0; % Start the optimization close to solution form cvx

lambda = 0.1;
eta = 0.1;

% All histories
alpha_hist = [alpha];
w_hist = [compute_w(alpha,y,x,lambda)];
dual_hist = [dual_objective(alpha, K, lambda)];
primal_hist = [primal_objective(y,compute_w(alpha,y,x,lambda) , x, lambda)];

num_iter = 500;

for i = 1:num_iter
    % Update
    grad = ((1/(2*lambda))*K*alpha)-1;
    d = -grad/norm(grad);
    alpha = alpha + eta*d;
    
    % Projection
    alpha = min(1, max(0, alpha));
    
    % Compute w
    w = compute_w(alpha, y, x, lambda);
    
    % Store all the histories
    alpha_hist = [alpha_hist alpha];
    w_hist = [w_hist w];
    
    primal_hist = [primal_hist primal_objective(y, w, x, lambda)];
    dual_hist = [dual_hist dual_objective(alpha, K, lambda)];
end

disp(['Solution with PGD: ' num2str(dual_hist(end))]);

% Plot the results 
figure(2)
plot3(w_hist(1,:), w_hist(2,:), primal_hist, 'color', 'r', 'lineWidth', 1.5);
legend('Hinge loss with regularization', 'Minimum point', 'PGD path')

figure(3);
clf;
hold on;
plot(0:num_iter, w_hist(1,:), 'linewidth', 2);
plot(0:num_iter, w_hist(2,:), 'linewidth', 2);
title('Optimization path for the weights with PGD')
xlabel('Iterations')
ylabel('Weight value')

legend('$w_1$', '$w_2$', 'interpreter', 'latex')

figure(5);
clf;
hold on; grid on;
plot(0:num_iter, primal_hist, 'linewidth', 2);
plot(0:num_iter, dual_hist, 'linewidth', 2);

title('Optimization path of primal and dual problem with PGD')
xlabel('Iterations')
ylabel('Objective function value')
legend('Primal problem objective value','Dual problem objective value')

figure(6);
clf;
hold on; grid on;
title('Duality Gap')
plot(0:num_iter, primal_hist - dual_hist, 'lineWidth', 2);
xlabel('Iteration')
ylabel('$p - d$', 'interpreter', 'latex')

%% 4. Investigation of optimization algorithms
addpath('C:\Users\hans\Documents\Tokyo Tech\Machine Learning\liblinear-2.30\windows');
addpath('C:\Users\hans\Documents\Tokyo Tech\Machine Learning\libsvm-3.23\windows');

% Dual coordinate optimization
options = '-s 13';
tic
model = train(y, sparse(x), [options]);
toc

% Sequential minimal optimization
options = '-s 2';
tic
model = svmtrain(y, sparse(x), [options]);
toc

%% Extra functions 
function w = compute_w(alpha, y, x, lambda)
    w = 1/(2*lambda)*x'*(alpha.*y);
end

function J = primal_objective(y, w, x, lambda)
    J = sum(max(0, 1.0 - y.*x*w)) + lambda*norm(w)^2;
end

function J = dual_objective(alpha, K, lambda)
    J = -1/(4*lambda)*alpha'*K*alpha + alpha'*ones(length(alpha),1);
end
