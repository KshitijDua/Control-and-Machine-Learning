function classification_NN_adam()
    %%%%%%% DATA %%%%%%%%%%%
    % xcoords, ycoords, targets
    x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
    x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
    y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];

    % Initialize weights and biases 
    rng(5000);
    W2 = 0.5*randn(2,2);
    W3 = 0.5*randn(3,2);
    W4 = 0.5*randn(2,3);
    b2 = 0.5*randn(2,1);
    b3 = 0.5*randn(3,1);
    b4 = 0.5*randn(2,1);

    % Adam hyperparameters
    beta1 = 0.9; % Momentum decay
    beta2 = 0.999; % Squared gradient decay
    epsilon = 1e-8; % Small constant for numerical stability
    alpha = 0.001; % Initial learning rate
    mW2 = zeros(size(W2));
    mW3 = zeros(size(W3));
    mW4 = zeros(size(W4));
    mb2 = zeros(size(b2));
    mb3 = zeros(size(b3));
    mb4 = zeros(size(b4));

    % Initialize squared gradient vectors for each parameter
    vW2 = zeros(size(W2));
    vW3 = zeros(size(W3));
    vW4 = zeros(size(W4));
    vb2 = zeros(size(b2));
    vb3 = zeros(size(b3));
    vb4 = zeros(size(b4));
    % Optimization loop with Adam
    Niter = 1e6;
    savecost = zeros(Niter, 1);
    for counter = 1:Niter
        k = randi(10);
        x = [x1(k); x2(k)];
        % Forward pass
        z2 = W2*x + b2;
        a2 = activate(z2);
        z3 = W3*a2 + b3;
        a3 = activate(z3);
        z4 = W4*a3 + b4;
        a4 = activate(z4);
        % Backward pass
        delta4 = diag(grad_activate(a4))*(a4 - y(:,k));
        delta3 = diag(grad_activate(a3))*(W4'*delta4);
        delta2 = diag(grad_activate(a2))*(W3'*delta3);
        % Adam update for each parameter
        mW2 = beta1 * mW2 + (1 - beta1) * (delta2 * x');
        mW3 = beta1 * mW3 + (1 - beta1) * (delta3 * a2');
        mW4 = beta1 * mW4 + (1 - beta1) * (delta4 * a3');
        mb2 = beta1 * mb2 + (1 - beta1) * delta2;
        mb3 = beta1 * mb3 + (1 - beta1) * delta3;
        mb4 = beta1 * mb4 + (1 - beta1) * delta4;

        vW2 = beta2 * vW2 + (1 - beta2) * (delta2 * x').^2;
        vW3 = beta2 * vW3 + (1 - beta2) * (delta3 * a2').^2;
        vW4 = beta2 * vW4 + (1 - beta2) * (delta4 * a3').^2;
        vb2 = beta2 * vb2 + (1 - beta2) * delta2.^2;
        vb3 = beta2 * vb3 + (1 - beta2) * delta3.^2;
        vb4 = beta2 * vb4 + (1 - beta2) * delta4.^2;

        mW2_hat = mW2 / (1 - beta1^counter);
        mW3_hat = mW3 / (1 - beta1^counter);
        mW4_hat = mW4 / (1 - beta1^counter);
        mb2_hat = mb2 / (1 - beta1^counter);
        mb3_hat = mb3 / (1 - beta1^counter);
        mb4_hat = mb4 / (1 - beta1^counter);

        vW2_hat = vW2 / (1 - beta2^counter);
        vW3_hat = vW3 / (1 - beta2^counter);
        vW4_hat = vW4 / (1 - beta2^counter);
        vb2_hat = vb2 / (1 - beta2^counter);
        vb3_hat = vb3 / (1 - beta2^counter);
        vb4_hat = vb4 / (1 - beta2^counter);

        W2 = W2 - alpha * mW2_hat ./ (sqrt(vW2_hat) + epsilon);
        W3 = W3 - alpha * mW3_hat ./ (sqrt(vW3_hat) + epsilon);
        W4 = W4 - alpha * mW4_hat ./ (sqrt(vW4_hat) + epsilon);
        b2 = b2 - alpha * mb2_hat ./ (sqrt(vb2_hat) + epsilon);
        b3 = b3 - alpha * mb3_hat ./ (sqrt(vb3_hat) + epsilon);
        b4 = b4 - alpha * mb4_hat ./ (sqrt(vb4_hat) + epsilon);

        % Monitor progress
        newcost = cost(W2, W3, W4, b2, b3, b4);
        savecost(counter) = newcost;
    end
figure(1)
clf
semilogy([1:1e4:Niter],savecost(1:1e4:Niter),'b-','LineWidth',2)
xlabel('Iteration Number')
ylabel('Value of cost function')
set(gca,'FontWeight','Bold','FontSize',18)
print -dpng pic_cost.png

%%%%%%%%%%% Display shaded and unshaded regions 
N = 500;
Dx = 1/N;
Dy = 1/N;
xvals = 0:Dx:1;
yvals = 0:Dy:1;
for k1 = 1:N+1
    xk = xvals(k1);
    for k2 = 1:N+1
        yk = yvals(k2);
        xy = [xk;yk];
         z2=W2*xy+b2;
         a2 = activate(z2);
         z3=W3*a2+b3;
         a3 = activate(z3);
         z4=W4*a3+b4;
         a4 = activate(z4);
        Aval(k2,k1) = a4(1);
        Bval(k2,k1) = a4(2);
     end
end

  function costval = cost(W2,W3,W4,b2,b3,b4)

     costvec = zeros(10,1); 
     for i = 1:10
         x =[x1(i);x2(i)];
         z2=W2*x+b2;
         a2 = activate(z2);
         z3=W3*a2+b3;
         a3 = activate(z3);
         z4=W4*a3+b4;
         a4 = activate(z4);
         costvec(i) = norm(y(:,i) - a4,2);
     end
     costval = norm(costvec,2)^2;
   end % of nested function
function y = activate(z)
 y = 1./(1+exp(-z)); 
end
    function y=grad_activate(x)
 y=x.*(1-x);
    end
end
