clc
clear all
clc
clear
%% Citirea si preprocesare datelor
dataFolder = 'train';
categories = {'Lotus', 'Tulip'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

dataFolder = 'test';
imds_t = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

%image_label=countEachLabel(imds_t)
%image_label=countEachLabel(imds)

data_t =[];                     % date de test
data=[];                        % date de antrenare     
while hasdata(imds) 
    img = read(imds) ;              % citeste o imagine din datastore
    img = imresize(img, [227 227]);
    img = double(rgb2gray(img));
    data =[data, reshape(img, [], 1)];
end
eticheta = double(imds.Labels == 'Tulip');% eticheta 1 - lalea, 0 - lotus

while hasdata(imds_t) 
    img = read(imds_t) ;              % citeste o imagine din datastore
    img = imresize(img, [227 227]); 
    img = double(rgb2gray(img));
    data_t =[data_t, reshape(img, [], 1)];
    
end
eticheta_t = double(imds_t.Labels == 'Tulip'); % eticheta 1 - lalea, 0 - lotus

%Reducerea dimensiuni
data_pca = pca(data, 'NumComponents', 45)';
data_pca_t = pca(data_t, 'NumComponents', 45)';

clear categories imds imds_t img dataFolder data data_t

 [n,N] = size(data_pca) ;    %A=data_pca
 m=10;                      %nr de neuroni de pe stratul ascuns
 x=rand(m,1);
 X=rand(n+1,m);
 
 I_N=ones(N,1);
 A=[data_pca',I_N];
 a=0.25;
 b=1;
 c=0.4;

% Metoda Gradient cu pas normal
 alpha=0.00003;
 iter=0;
 epsilon=10^(-5);
 maxIter=500;
 t_const=zeros(1,maxIter);
 err=[];

criteriu= 10^4;
L_old=criteriu;

tic
while (all(criteriu > eps) && iter < maxIter)

    iter=iter+1;
    disp(iter);

    g=zeros(1,N);
    for i=1:N
        z = A(i, :) * X;  % X ar trebui să fie o matrice de greutăți corespunzătoare întregii matrice A
        activation = VFS(z, a, b, c);
        g(i) = mean(activation);
    end
       
% Calculăm mai întâi toate gradienții necesari
    grad_g = grad(A, X, a, b, c);  % Presupunem că A este nxN și X este nxm, ajustează dimensiunile conform necesităților
    grad_loss = grad_L(eticheta, g, a, b, c, N);  % g ar trebui să fie vectorul de activări
    %[grad_X,grad_x]=compute_gradients(X, x, A, eticheta, a, b, c)

    gradx=grad_x(eticheta,A,X,x,a,b,c);
    gradX=gradL_X(eticheta,A,X,x,a,b,c);

    % Actualizarea greutăților
    X = X - alpha * gradX;
    x = x - alpha * gradx;

% Asigură-te că dimensiunile permit o înmulțire scalară
    L_current = L(eticheta,VFS(A*X,a,b,c)*x,N)  % Ajustează această linie conform logicii tale specifice
    
    criteriu=norm(L_current - L_old)
    err=[err,criteriu];
    L_old = L_current;

   
end
t_const=toc;

disp(t_const)
figure
semilogy(err);
title("Convergetnta")

% Metoda Gradient stocastic 
% Inițializări
[n,N]=size(data_pca)
[n,N] = size(data_pca) ;    %A=data_pca
 m=10;                      %nr de neuroni de pe stratul ascuns
 x=rand(m,1);
 X=rand(n+1,m);
 
 I_N=ones(N,1);
 A=[data_pca',I_N];
 a=0.25;
 b=1;
 c=0.4;

rataInvatare = 0.00003;  % Rata de învățare
dimensiuneBatch = 50;  % Mărimea fiecărui mini-batch
numarBatchuri = floor(N / dimensiuneBatch);  % Numărul de batches
iteratiiMaxime = 500;  % Numărul maxim de iterații
 err2=[];

% Ciclul de antrenament SGD
for iteratie = 1:iteratiiMaxime
    % Amestecarea datelor
    indici = randperm(N);
    A_amestecat = A(indici, :);
    eticheta_amestecat = eticheta(indici);
    
    for batch = 1:numarBatchuri
        % Selectarea unui mini-batch
        indexInceput = (batch - 1) * dimensiuneBatch + 1;
        indexSfarsit = batch * dimensiuneBatch;
        A_miniBatch = A_amestecat(indexInceput:indexSfarsit, :);
        eticheta_miniBatch = eticheta_amestecat(indexInceput:indexSfarsit);
        
        % Calculul gradientului pentru mini-batch
        gradX_miniBatch = gradL_X(eticheta_miniBatch, A_miniBatch, X, x, a, b, c);
        gradx_miniBatch = grad_x(eticheta_miniBatch, A_miniBatch, X, x, a, b, c);
        
        % Actualizarea greutăților
        X = X - rataInvatare * gradX_miniBatch;
        x = x - rataInvatare * gradx_miniBatch;
    end
    
    % O opțiune: Calculează și afișează pierderea la fiecare epocă
    pierdereCurenta = L(eticheta, VFS(A * X, a, b, c) * x, N);
    fprintf('Iterația %d, Pierdere: %f\n', iteratie, pierdereCurenta);
    err2=[err2,pierdereCurenta];
end

figure
semilogy(err);
title("Convergetnta MG stocatic");

% Matrice de confuzie

% Predicțiile modelului pe setul de test
Z_test = data_pca_t .* X + repmat(x, size(data_pca_t, 1), 1);
Y_pred = VFS(Z_test, a, b, c);

% Trebuie să transformăm ieșirile continue în clase binare (0 sau 1)
% Presupunem că folosim 0.5 ca prag
clase_pred = Y_pred > 0.5;
% Calculul matricei de confuzie
matrice_confuzie = confusionmat(eticheta_t, clase_pred);

% Afișarea matricei de confuzie
disp('Matricea de confuzie:');
disp(matrice_confuzie);

figure;
confusionchart(matrice_confuzie, {'Lotus', 'Tulip'});
title('Matrice de Confuzie pentru Clasificarea Florilor');


