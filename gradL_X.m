function [gradX]=grad_X(e,A,X,x,a,b,c)
   
[N,n]=size(A);
[~,m]=size(X);
gradX=zeros(n,m);

for k=1:n
    sum=0;
    for i=1:N
        sum=sum+e(i)*(1/(VFS(A(i,:)*X,a,b,c)*x))*...
           grad(A(i,:),X,a,b,c)  * A(i,k).*x' +...
        (1-e(i))*(1/(1-VFS(A(i,:)*X,a,b,c)*x))*...
            grad(A(i,:),X,a,b,c) * A(i,k).*x';
    end
    gradX(k,:)=-sum./N;
end