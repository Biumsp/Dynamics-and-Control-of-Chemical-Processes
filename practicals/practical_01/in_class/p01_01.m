% Enrico Bussetti
% Dynamics and Control of Chemical Processes
% Practical 1, exercise 1

close all
clear variables

format compact
format short g

% -------------------------------------------------------------------------
% Data
% -------------------------------------------------------------------------

k1 = 0.5;       % [h^-1]
k2 = 1e-7;      % [kmol/m^3/h]
k3 = 0.6;       % [-]

B0 = 0.03;      % [kmol/m^3]
S0 = 4.5;       % [kmol/m^3]

t_end = 15;

% -------------------------------------------------------------------------
% Solution
% -------------------------------------------------------------------------

opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-12);

% Initial conditions
IC = [B0, S0];

[t, y] = ode15s(@function_1, [0, t_end], IC, opts, ...
                 k1, k2, k3);
             
B = y(:, 1);
S = y(:, 2);


% -------------------------------------------------------------------------
% Graphical-Post-Processing
% -------------------------------------------------------------------------

figure(1)
plot(t, B, 'r', t, S, 'b', 'LineWidth', 2)
set(gca, 'Fontsize', 14)
legend('Biomass', 'Substrate', 'Location','best')
xlabel('Time [h]')
ylabel('Concentration [kmol/m^3]')

% -------------------------------------------------------------------------
% Functions
% -------------------------------------------------------------------------

function yy = function_1(t, var, k1, k2, k3)
    
    B = var(1);
    S = var(2);
    
    dB =     k1*B*S/(k2 + S);
    dS = -k3*k1*B*S/(k2 + S);
    
    yy = [dB dS]';

end
