function [gradx]=grad_x(e,A,X,x,a,b,c)

[N,n]=size(A);
[~,m]=size(X);
gradx=zeros(m,1);

for k=1:m
    sum=0;
    for i=1:N
        sum=sum+e(i)*(1/( VFS(A(i,:)*X,a,b,c)*x ))*...
        VFS(A(i,:)*X(:,k),a,b,c)+(1-e(i))*(1/(1-VFS(A(i,:)*X,a,b,c)*x))*VFS(A(i,:)*X(:,k),a,b,c);
    end
    gradx(k)=-sum/N;
end



end