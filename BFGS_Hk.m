function quasi_newton_methods_BFGS_Hk()

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
    

    Hk = eye(2); 
    
    for k = 1:max_iterations
        grad = gradient(x);
        
        % Stopping criterion
        if norm(grad) < epsilon
            break;
        end
        
        % Search direction
        dk = -Hk * grad;
        
        % Backtracking line search
        t = 1;
        while f(x + t * dk) > f(x) + alpha * t * grad' * dk
            t = beta * t;
        end
        
        % Update x
        x_next = x + t * dk;
        
        % Update gradient
        grad_next = gradient(x_next);
        
        % Update Hessian approximation using BFGS formula
        sk = x_next - x;
        yk = grad_next - grad;
        rho = 1 / (yk' * sk);
        Hk = (eye(2) - rho * sk * yk') * Hk * (eye(2) - rho * yk * sk') + rho * sk * sk';
        
        % Update x and gradient for next iteration
        x = x_next;
        grad = grad_next;
    end
    
    % Results
    fprintf('Optimization finished after %d iterations.\n', k);
    fprintf('Optimal solution: x = [%f, %f]\n', x(1), x(2));
    fprintf('Objective function value at optimal solution: f(x) = %f\n', f(x));

end
