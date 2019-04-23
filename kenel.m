clc;clear all;
rng(20)%保证每次生成的随机数不变
addpath(genpath('.\libsvm'))
mu1=[2,3];
SIGMA1=[1.2,0;0,2];
mu2=[8,9];
SIGMA2=[1.2,0;0,2];
mu3=[2,9];
SIGMA3=[1.2,0;0,2];
mu4=[8,3];
SIGMA4=[1.2,0;0,2];
data1=mvnrnd(mu1,SIGMA1,500);%mvnrnd 是用来生成多维正态数据的
data2=mvnrnd(mu2,SIGMA2,500);
data3=mvnrnd(mu3,SIGMA3,500);
data4=mvnrnd(mu4,SIGMA4,500);
data=[data1;data2;data3;data4];
figure(1);
plot(data1(:,1),data1(:,2),'r+');
hold on;
plot(data2(:,1),data2(:,2),'r+');
hold on;
plot(data3(:,1),data3(:,2),'g*');
hold on;
plot(data4(:,1),data4(:,2),'g*');
title('原始数据');

%给数据加label
index1=randperm(500);
index2=randperm(500);
index3=randperm(500);
index4=randperm(500);
trX1=[data1(index1(1:250),:);data2(index2(1:250),:)];
%trlabels=[ones(500,1);2*ones(500,1)];
trX2=[data3(index3(1:250),:);data4(index4(1:250),:)];
trlabels=[ones(500,1);2*ones(500,1)];

teX1=[data1(index1(251:end),:);data2(index2(251:end),:)];
%telabels=[ones(500,1);2*ones(500,1)];
teX2=[data3(index3(251:end),:);data4(index4(251:end),:)];
telabels=[ones(500,1);2*ones(500,1)];

trX=[trX1;trX2];
teX=[teX1;teX2];

figure(2);
Struct1=svmtrain(trX,trlabels,'Kernel_Function','quadratic','showplot',true);
classes1=svmclassify(Struct1,teX);
%classes1=svmclassify(Struct1,teX,'showplot',true);
title('二次核函数');
CorrectRate1=sum(trlabels==classes1)/1000 %表示分类正确率

figure(3);
Struct2=svmtrain(trX,trlabels,'showplot',true);
classes2=svmclassify(Struct2,teX);
title('线性核函数');
CorrectRate2=sum(trlabels==classes2)/1000

figure(4);
Struct2=svmtrain(trX,trlabels,'Kernel_Function','rbf','RBF_Sigma',1,'showplot',true);
classes2=svmclassify(Struct2,teX);
title('高斯径向基核函数(核宽1)');
CorrectRate2=sum(trlabels==classes2)/1000

figure(5);
Struct2=svmtrain(trX,trlabels,'Kernel_Function','rbf','RBF_Sigma',3,'showplot',true);
classes2=svmclassify(Struct2,teX);
title('高斯径向基核函数(核宽3)');
CorrectRate2=sum(trlabels==classes2)/1000

figure(6);
Struct3=svmtrain(trX,trlabels,'Kernel_Function','polynomial','showplot',true);
classes3=svmclassify(Struct3,teX);
title('多项式核函数');
CorrectRate3=sum(trlabels==classes3)/1000

figure(7);
Struct4=svmtrain(trX,trlabels,'Kernel_Function','mlp','showplot',true);
classes4=svmclassify(Struct4,teX);
title('多层感知机核函数');
CorrectRate4=sum(trlabels==classes4)/1000
