function [grad_X, grad_x] = compute_gradients(X, x, A, t, a, b, c)

    
    % Calculul ieșirii rețelei
    Z = A * X + x';
    Y = VFS(Z, a, b, c);
    
    % Derivata funcției VFS
    sigmoidZ = 1 ./ (1 + exp(-b * Z));
    dVFS = a * b * sigmoidZ .* (1 - sigmoidZ);  % Derivata VFS în raport cu Z
    
    % Eroarea
    E = Y - t;  % t sunt etichetele reale
    
    % Calculul gradientului pentru X și x
    grad_X = A' * (E .* dVFS);  % înmulțire matriceală pentru gradientul lui X
    grad_x = sum(E .* dVFS, 1);  % sumă pe coloane pentru gradientul lui x
end