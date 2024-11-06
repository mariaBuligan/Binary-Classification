function [v] = grad_L(e,y,a,b,c,N)
    v = zeros(1, N);

    
    for i = 1:N
        v(i) = (-e(i) / y(i)) + ((1 - e(i)) / (1 - y(i)));
    end
    v = sum(v) / N;  % Media gradientului
       
    % L'=[ -1/N * sum(log(y(i)) +1/N * sum(log(1-y(i)) ;    %in raport cu e
 
    %  -1/N * sum(e(i)/y(i)) +1/N * sum( 1-e(i)/1-y(i) ) ] % in raport cu y
end