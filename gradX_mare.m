function [gradX]=gradX_mare(e,A,X,x,a,b,c,N)
a=3;
[N,n]=size(A);
[~,m]=size(X);
gradX=zeros(n,m);

for k=1:n
    sum=0;
    for i=1:N
       % z=A(i,:)*X;
       % z_deriv=A(i,k);

        sum=sum+e(i)*(1/(VFS(A(i,:)*X,a,b,c)*x))*gradL_X(e,A,X,x,a,b,c).*x'+(1-e(i))*(1/(1-VFS(A(i,:)*X,a,b,c)*x))*gradL_X(e,A,X,x,a,b,c).*x';
    end
    
    gradX(k,:)=-sum/N;
end
end