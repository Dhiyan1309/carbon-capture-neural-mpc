clear;
clc;
close all;
k=1.5;
tau=3.8*3600;
theta=15*60;

s=tf('s');
plant=(k/(s*tau+1))*exp(-s*theta);

kp=1.2;
ki=0.005;
kd=0;
controller=pid(kp,ki,kd);
closedloop=feedback(controller*plant,1);

t=0:10:(15*3600);
[y,tout]=step(closedloop,t);

figure;
plot(tout/3600,y,'r','LineWidth',2);
title('PID Controller Failure on High Latency CCS Plant');
xlabel('Time(Hours)');
ylabel('Temperature Change(Delta T)');
grid on;
yline(1, 'k--', 'Target Reference', 'LineWidth', 1.5);
