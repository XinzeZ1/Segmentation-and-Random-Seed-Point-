close all
clear
clc

%Load Data
mask = 

mask = load_untouch_nii('AAAA_2007.01.21_LPS_rT1ce_segmdm.nii.gz');
mask_img = mask.img;
mask_final = int16(double(logical(double(mask_img))));
img = load_untouch_nii('AAAA_2007.01.21_t2_LPS_rT1ce_SSFinal.nii.gz');
back_img = img.img;
[~, ~, Q] = ind2sub( size(back_img), find(back_img));
mq1 = min(Q);
mq2 = max(Q);
aver = ceil((mq1+mq2)/2);
high = aver+20;
low = aver-20;
tic 
for tl = low:high
bw2 = double((back_img(:,:,tl)));
Seg_img = bw2.*~(double(mask_final(:,:,tl)));



%Segmentation Part
I=Seg_img;
[m,n]=size(I);           %图像分辨率为m*n
ROI=I>0;
ROI=double(ROI);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------initilization--------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Step 1: Initilization
K=3;%K为分类数
epsilon=0.1;
iterNum=100;

Mu=zeros(K,1);            % Mu为均值向量
Sigma=zeros(K,1);         % 方差
PI=zeros(m,n,K);          % PI为先验分布
Z=zeros(m,n,K);           % 后验概率
F=zeros(m,n,K);           % 空间因子
B=ones(m,n);              % bias field

radius=1;                 % 矩形邻域半径
beta=1;                 % 温度值
h=10;                    % h用于控制空间权重因子
W=zeros(m,n,9);
w=zeros(3,3);
for i=3:m-2
     for j=3:n-2
          for x=-1:1
               for y=-1:1
                    window=I(i+x-1:i+x+1,j+y-1:j+y+1);
                    w(x+2,y+2)=exp(-sum(sum(abs(window-I(i-1:i+1,j-1:j+1))))/h);
               end
               W(i,j,:)=w(:);
          end
    end
end
%计算空间权重系数W

%basis funtions for bias field estimation
demination = 10;
basis_only_phik = getBasisOrder3(m,n);
A = basis_only_phik;

%K-means for initialization
yy=I(ROI==1);
yy=sort(yy,'descend');
[IDX,C] = kmeans(yy,K,'Start','cluster','Maxiter',100,'EmptyAction','drop','Display','off');
while sum(isnan(C))>0
[IDX,C] = kmeans(yy,K,'Start','cluster','Maxiter',100,'EmptyAction','drop','Display','off');
end
Mu=sort(C);
Dis_k=zeros(m,n,K);
for k=1:K
    Dis_k(:,:,k)=(I-Mu(k)).^2;
end
for k=1:K
    [e_min,IDX]=min(Dis_k,[],3);
    IDX_ROI=IDX.*ROI;
    Sigma(k)=var(I(IDX_ROI==k));
    PI(1:m,1:n,k) = sum(sum(double(IDX_ROI==k)))/(sum(ROI(:)));
end
Mu_old =Mu;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------Iteration----------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iteration_times=1:iterNum
   % Step 2: E-step
    %Evaluate the posterior probabilities Z
    for k = 1:K
        Z(:,:,k)=PI(:,:,k).*exp(-0.5*(I-Mu(k)*B).^2.*(1/(Sigma(k)+eps)))./(sqrt(2*pi*(Sigma(k)+eps)));
        Z(:,:,k)=Z(:,:,k).*ROI;
    end
    Z_sum=sum(Z,3);
    for k = 1:K
        Z(:,:,k) = Z(:,:,k)./(Z_sum+eps).*ROI;
    end
    
   %update the factor F
   for k = 1:K
        for i=2:m-1
            for j=2:n-1
                window=Z(i-1:i+1,j-1:j+1,k)+PI(i-1:i+1,j-1:j+1,k);
                w=reshape(W(i,j,:),3,3);
                F(i,j,k)=exp(beta*sum(sum(w.*window))/(sum(sum(w))+eps));
            end
        end
   end
    
   %Step 3: M-step
   %Update the parameter vector (Mu & Sigma) and pixel label priors
    for k = 1:K
        Mu(k)=sum(sum(Z(:,:,k).*I.*B))/sum(sum(Z(:,:,k).*(B.^2)+eps));
        Sigma(k)=sum(sum(Z(:,:,k).*(I-Mu(k).*B).^2))/sum(sum(Z(:,:,k)+eps));
        PI(:,:,k)=(Z(:,:,k) + F(:,:,k)).*ROI;
    end
    PI_sum=sum(PI,3);
    for k = 1:K
        PI(:,:,k)=PI(:,:,k)./(PI_sum + eps).*ROI;
    end
    
   % update w
        J1 = zeros(m,n);
        J2 = zeros(m,n);
        for k = 1:K
        J1 = J1 + Z(:,:,k)*Mu(k)/(2*Sigma(k));
        J2 = J2 + Z(:,:,k)*Mu(k)^2/(2*Sigma(k));
        end
        J1 = J1.*ROI;J2 = J2.*ROI;
        v = zeros(demination,1);
        for k = 1:demination
        v(k) = sum(sum(I.*A(:,:,k).*J1.*ROI));
        end
        AA = zeros(demination,demination);
        for k = 1:m
        for kk = 1:n
            temp = reshape(A(k,kk,:),demination,1);
            AA = AA + temp*temp'*J2(k,kk);
        end
        end
        w = inv(AA)*v;
   % update bias
        temp = zeros(m,n);
        for i = 1:demination
        temp = temp + w(i).*A(:,:,i);
        end
        B = temp.*ROI + (1-ROI);
    
    [e_max,N_max]=max(Z,[],3);
    Img_out=N_max;
    
    iterNums=['segmentation: ',num2str(iteration_times), ' iterations'];
    subplot(1,2,1),imshow(I,[]),title('original')
    subplot(1,2,2),imshow(Img_out.*ROI,[]),title(iterNums); colormap(gray);
    pause(0.1)
    
   % check the convergence
    compare=sqrt(sum(sum((Mu_old/255 - Mu/255).^2)));
    if compare<epsilon
        break;
    else
        Mu_old = Mu;
    end
end
%迭代结束
[Mu_temp,Mu_IX]=sort(Mu);
for k=1:K
    Mu_temp(Mu_IX(k))=k;
end

img_out=zeros(m,n);
for i=1:m
    for j=1:n
        max_U=Z(i,j,1);
        x = 1;
        for k=2:K
            if max_U<Z(i,j,k)
               max_U=Z(i,j,k);
               x=k;
            end
        end
        img_out(i,j)=Mu_temp(x);
    end
end

c=zeros(m,n);
e=(ones(m,n).*ROI)/sum(sum(ROI));
U=zeros(m,n,K);
for i_iteration=1:10
    for k=1:K
         U(:,:,k)=Z(:,:,k).*log2(Z(:,:,k)./(sum(sum(Z(:,:,k).*e)))+eps);
         U(:,:,k)=U(:,:,k).*ROI;
    end
    c=exp(sum(U,3));
    c=c.*ROI;
    %计算c
    e=(e.*c)/(sum(sum(c))+eps);
    e=e.*ROI;
   % 计算e
end
Xi=zeros(K,1);
cluster_label=img_out;
for i=2:m-1
    for j=2:n-1
       if e(i,j)>0&&e(i,j)<0.005*10^(-46)
            window=cluster_label(i-1:i+1,j-1:j+1);
            for k=1:K
                Xi(k)=sum(window(:)==k);
            end
            [maxcluster,img_out(i,j)]=max(Xi);
       end
    end
end
%考虑独立噪声点

img_out=img_out.*ROI;
IM=img_out*50;   
imwrite(uint8(IM),['AAAN_2007.03.11_',num2str(tl-1),'.bmp']);
end
toc

A1=[];B1=[];R1=[];C1=[];
A2=[];B2=[];R2=[];C2=[];
A3=[];B3=[];R3=[];C3=[];
num_slice = high-low;
K1=[];K2=[];K3=[];
MAT_label1 = [];
MAT_label2 = [];
MAT_label3 = [];
R_MAT_label1 = [];
R_MAT_label2 = [];
R_MAT_label3 = [];
for ti=low:high
%%Get Coordinate
IM=imread(['AAAN_2007.03.11_',num2str(ti-1),'.bmp']);
FS=double(IM);
 j1 = numel(find(FS==50));%%%(CSF==50;GM==100;WM==150)
 j2 = numel(find(FS==100));
 j3 = numel(find(FS==150));
 
     k1 = floor(j1*0.01);
     K1 = [K1;k1];
     k2 = floor(j2*0.01);
     K2 = [K2;k2];
     k3 = floor(j3*0.01);
     K3 = [K3;k3];
     if k1~=0
     atlas_label1 = (FS==50);%%%
     end
     if k2~=0
     atlas_label2 = (FS==100);
     end
     if k3~=0
     atlas_label3 = (FS==150);
     end
      
     % Mark Foreground and Get Point Coordinates
     [L1,num1] = bwlabel(atlas_label1);
     [L2,num2] = bwlabel(atlas_label2);
     [L3,num3] = bwlabel(atlas_label3);
     
     % Getting Coordinates of All Foreground Pixels
     [r1,c1] = find(atlas_label1==1);
     [r2,c2] = find(atlas_label2==1);
     [r3,c3] = find(atlas_label3==1);
     
     rr1 = r1';
     R1 = [R1,rr1];
     cc1 = c1';
     C1 = [C1,cc1];
     
     rr2 = r2';
     R2 = [R2,rr2];
     cc2 = c2';
     C2 = [C2,cc2];
     
     rr3 = r3';
     R3 = [R3,rr3];
     cc3 = c3';
     C3 = [C3,cc3];
     
     % Random Acquisition of Sequence Number and Random Point Layout
     r_xulie1 = randi([1 length(r1)],1,k1);
     r_xulie2 = randi([1 length(r2)],1,k2);
     r_xulie3 = randi([1 length(r3)],1,k3);
     
     %Get Coordinate Data and Draw Points
     xx1 = r1(r_xulie1);
     yy1 = c1(r_xulie1);
     x1 = xx1';
     A1 = [A1,x1];
     y1 = yy1';
     B1 = [B1,y1];
     
     xx2 = r2(r_xulie2);
     yy2 = c2(r_xulie2);
     x2 = xx2';
     A2 = [A2,x2];
     y2 = yy2';
     B2 = [B2,y2];
     
     xx3 = r3(r_xulie3);
     yy3 = c3(r_xulie3);
     x3 = xx3';
     A3 = [A3,x3];
     y3 = yy3';
     B3 = [B3,y3];
       
     z_vec1 =(ti-1)*ones(size(xx1));
     vec1 = [xx1, yy1, z_vec1];
     
     z_vec2 =(ti-1)*ones(size(xx2));
     vec2 = [xx2, yy2, z_vec2];
          
     z_vec3 = (ti-1)*ones(size(xx3));
     vec3 = [xx3, yy3, z_vec3];
     
     
     MAT_label1 = [MAT_label1; vec1];
     MAT_label2 = [MAT_label2; vec2];
     MAT_label3 = [MAT_label3; vec3];
     
     [m1,n1] = size(MAT_label1);
     [m2,n2] = size(MAT_label2);
     [m3,n3] = size(MAT_label3);
     
     total_k = k1+k2+k3;
     ratio_k1 = floor((k1/total_k)*40);
     ratio_k2 = floor((k2/total_k)*40);
     ratio_k3 = floor((k3/total_k)*40);
     
     R_MAT_label1 = MAT_label1(randperm(m1, ratio_k1),:);
     R_MAT_label2 = MAT_label2(randperm(m2, ratio_k2),:);
     R_MAT_label3 = MAT_label3(randperm(m3, ratio_k3),:);
               
end
xlswrite('AAAN_2007.03.11.xlsx',R_MAT_label1,1)
xlswrite('AAAN_2007.03.11.xlsx',R_MAT_label2,2)
xlswrite('AAAN_2007.03.11.xlsx',R_MAT_label3,3)
dlmwrite('AAAN_2007.03.11.txt',R_MAT_label1,'delimiter','\t')
dlmwrite('AAAN_2007.03.11.txt',R_MAT_label2,'-append','delimiter','\t','roffset',1)
dlmwrite('AAAN_2007.03.11.txt',R_MAT_label3,'-append','delimiter','\t','roffset',2)

function [B] = getBasisOrder3(Height,Wide)

for i =1:Height
    x(i,:) = -1:2/(Wide-1):1;
end
for i =1:Wide
    temp = -1:2/(Height-1):1;
    y(:,i) = temp';
end

bais = zeros(Height,Wide,10);
bais(:,:,1) = 1;
bais(:,:,2) = x;
bais(:,:,3) = (3.*x.*x - 1)./2;
bais(:,:,4) = (5.*x.*x.*x - 3.*x)./2;
bais(:,:,5) = y;
bais(:,:,6) = x.*y;
bais(:,:,7) = y.*(3.*x.*x -1)./2;
bais(:,:,8) = (3.*y.*y -1)./2;
bais(:,:,9) = (3.*y.*y -1).*x./2;
bais(:,:,10) = (5.*y.*y.*y -3.*y)./2;

B = bais;
for kk=1:10
    A=B(:,:,kk).^2;
    r = sqrt(sum(A(:)));
    B(:,:,kk)=B(:,:,kk)/r;
end
end