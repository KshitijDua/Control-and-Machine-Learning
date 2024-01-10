function quasi_newton_methods_DFP_Bk()

    % Parameters
    max_iterations = 50000;
    epsilon = 1e-8;
    alpha = 0.25;
    beta = 0.5;

    % Initial values
    x = [0; 0];
    
    % Function definition
    f = @(x) 100 * (x(2) - x(1)^2)^2 + (x(1) - 1)^2;
    gradient = @(x) [400 * x(1)^3 - 400 * x(1) * x(2) + 2 * x(1) - 2; 200 * (x(2) - x(1)^2)];
    
    
    Bk = eye(2); % Initial inverse Hessian approximation
    
    for k = 1:max_iterations
        grad = gradient(x);
        
        % Stopping criterion
        if norm(grad) < epsilon
            break;
        end
        
        % Search direction
        dk = -Bk * grad;
        
        % Backtracking line search
        t = 1;
        while f(x + t * dk) > f(x) + alpha * t * grad' * dk
            t = beta * t;
        end
        
        % Update x
        x_next = x + t * dk;
        
        % Update gradient
        grad_next = gradient(x_next);
        
        % Update Hessian approximation using DFP formula
        sk = x_next - x;
        yk = grad_next - grad;
        
        Bk = Bk + (sk * sk') / (sk' * yk) - (Bk * yk * yk' * Bk) / (yk' * Bk * yk);
        
        % Update x and gradient for next iteration
        x = x_next;
        grad = grad_next;
    end
    
    % Results
    fprintf('Optimization finished after %d iterations.\n', k);
    fprintf('Optimal solution: x = [%f, %f]\n', x(1), x(2));
    fprintf('Objective function value at optimal solution: f(x) = %f\n', f(x));

end
