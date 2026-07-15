clear;clc;
k=1.5;tau=3.8*3600;theta=15*60;
s=tf('s');
plant=(k/(s*tau+1))*exp(-theta*s);

ts=60;
discrete_plant=c2d(plant,ts,'zoh');
num_samples=40000;
t=(0:num_samples-1)'*ts;
u=zeros(num_samples,1);
current_valve=50;

for i=1:num_samples
    if mod(i,60)==0
        current_valve=randi([20,100]);
    end
    u(i)=current_valve;
end

[y,~]=lsim(discrete_plant,u,t);

snr_db = 40;                                      % Target SNR
signal_power = sum(y.^2) / length(y);             % Measure the signal's power
noise_power = signal_power / (10^(snr_db / 10));  % Calculate required noise power
y_noisy = y + sqrt(noise_power) * randn(size(y)); % Generate and add Gaussian noise
dataset = table(t, u, y_noisy, 'VariableNames', {'Time_sec', 'Valve_Cmd', 'Temp_Response'});
writetable(dataset, 'ccs_training_data.csv');

disp('SUCCESS: 27 days of simulated plant data saved to ccs_training_data.csv!');
