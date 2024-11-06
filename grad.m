function [g] = grad(A,X,a,b,c)

    Z = A * X;  % Z este inputul în sigmoid
    sigmoidZ = sigmoid(b * Z);
    g = a * b * sigmoidZ .* (1 - sigmoidZ);  % .* pentru operația element cu element

        
end