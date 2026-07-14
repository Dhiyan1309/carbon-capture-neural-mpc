% System Identification for Carbon Capture Reboiler
% Simulating First-Order Plus Dead Time (FOPDT)

K = 1.2;              % Process Gain
tau = 228;            % Time Constant (approx 3.8 hours in minutes)
dead_time = 15;       % Transport Delay (minutes)

s = tf('s');
plant_tf = (K / (tau * s + 1)) * exp(-dead_time * s);

t = (0:1:1000)';      % Time vector      
[y, t] = step(plant_tf, t);

figure;
plot(t, y, 'LineWidth', 2);
title('Carbon Capture Thermal Inertia (Step Response)');
xlabel('Time (Minutes)');
ylabel('Temperature Shift');
grid on;

% EXPORT THE DATASET FOR AI TRAINING
dataset = table(t, y, 'VariableNames', {'Time_Minutes', 'Temperature'});
writetable(dataset, '../python_ai_controller/carbon_capture_data.csv');
disp('SUCCESS: Dataset exported to carbon_capture_data.csv');
