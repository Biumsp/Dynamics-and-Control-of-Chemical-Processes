% Enrico Bussetti
% Dynamics and Control of Chemical Processes
% Practical 1, exercise 2

close all
clear variables

format compact
format short g

% -------------------------------------------------------------------------
% Data
% -------------------------------------------------------------------------

Q = 1e3;        % [kW]
F = 8;          % [kmol/s]
m = 100;        % [kmol]

cp  = 2.5;      % [kJ/kmol/K]
Tin = 300;      % [K]
T_dev = 30;     % [K]

% Steady-State solution for Tout (we work with deviation variables)
Tout_ss = Q/F/cp + Tin;
fprintf('Steady-State Tout = %f [K]\n', Tout_ss)

% Time at which the deviation occurs
t_dev = 150;    % [s]

% End time 
t_end = 300;    % [s]

% -------------------------------------------------------------------------
% Solution
% -------------------------------------------------------------------------

opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-12);

% Initial conditions
IC1 = Tin;
[t1, T1] = ode15s(@function_1, [0, t_dev], IC1, opts, ...
                 Q, F, cp, Tin, m);
             
IC2 = T1(end);
[t2, T2] = ode15s(@function_1, [t_dev, t_end], IC2, opts, ...
                 Q, F, cp, Tin + T_dev, m);

% -------------------------------------------------------------------------
% Graphical-Post-Processing
% -------------------------------------------------------------------------

figure(1)
plot(t1, T1, 'b', t2, T2, 'LineWidth', 2)
set(gca, 'Fontsize', 14)
legend('Temperature','Location','best')
xlabel('Time [s]')
ylabel('Temperature [K]')
ylim([300 400])
title('Stirred Tank Heater dynamics')

% -------------------------------------------------------------------------
% Functions
% -------------------------------------------------------------------------

function yy = function_1(t, T, Q, F, cp, Tin, m)
    
    yy = Q/m/cp - F/m*(T - Tin);

end
