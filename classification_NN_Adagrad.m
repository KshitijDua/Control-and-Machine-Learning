function classification_NN_Adagrad
%%%%%%% DATA %%%%%%%%%%%
% xcoords, ycoords, targets
x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize weights and biases 
rng(5000);
W2 = 0.5*randn(2,2);
W3 = 0.5*randn(3,2);
W4 = 0.5*randn(2,3);
b2 = 0.5*randn(2,1);
b3 = 0.5*randn(3,1);
b4 = 0.5*randn(2,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Adagrad Initialization
G_W2 = zeros(size(W2));
G_W3 = zeros(size(W3));
G_W4 = zeros(size(W4));
G_b2 = zeros(size(b2));
G_b3 = zeros(size(b3));
G_b4 = zeros(size(b4));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward and Back propagate 
% Pick a training point at random
eta = 0.4;
Niter = 1e6;
eps = 1e-8;
savecost = zeros(Niter,1);
for counter = 1:Niter
    k = randi(10);
    x = [x1(k); x2(k)];
    % Forward pass
         z2=W2*x+b2;
         a2 = activate(z2);
         z3=W3*a2+b3;
         a3 = activate(z3);
         z4=W4*a3+b4;
         a4 = activate(z4);
    % Backward pass
    delta4 = diag(grad_activate(a4))*(a4-y(:,k));
    delta3 = diag(grad_activate(a3))*(W4'*delta4);
    delta2 = diag(grad_activate(a2))*(W3'*delta3);
    % RMSprop update
    G_W2 = G_W2 + ((delta2 * x').^2);
    G_W3 = G_W3 + ((delta3 * a2').^2);
    G_W4 = G_W4 +((delta4 * a3').^2);
    G_b2 = G_b2 + delta2.^2;
    G_b3 = G_b3 + delta3.^2;
    G_b4 = G_b4 + delta4.^2;
    
    W2 = W2 - (eta ./ sqrt(G_W2 + eps)) .* (delta2 * x');
    W3 = W3 - (eta ./ sqrt(G_W3 + eps)) .* (delta3* a2');
    W4 = W4 - (eta ./ sqrt(G_W4 + eps)) .* (delta4* a3');
    b2 = b2 - (eta ./ sqrt(G_b2 + eps)) .* delta2;
    b3 = b3 - (eta ./ sqrt(G_b3 + eps)) .* delta3;
    b4 = b4 - (eta ./ sqrt(G_b4 + eps)) .* delta4;
    
    % Monitor progress
    newcost = cost(W2,W3,W4,b2,b3,b4);   % display cost to screen
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
