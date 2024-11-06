function err = L(e,y,N)

entropie=0;
    for i=1:N
        entropie = entropie +e(i)*log(y(i))+(1-e(i))*log(1-y(i));
    end
    
    err=-entropie/N;

end