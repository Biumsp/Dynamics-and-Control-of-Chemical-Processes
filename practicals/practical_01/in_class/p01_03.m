% Enrico Bussetti
% Dynamics and Control of Chemical Processes
% Practical 1, exercise 3

close all
clear variables

format compact
format short g

% -------------------------------------------------------------------------
% Data
% -------------------------------------------------------------------------

C1 = 0.5;       % [kmol/m^3]
C2 = 6;         % [kmol/m^3]

% Initial value of F1
F1 = 2;         % [kmol/h]
F2 = 10;        % [kmol/h]

V = 1;          % [m^3]

% Steady-State (initial) values. Computed from the ss balances
Fout = F1 + F2;
Cout = (C1*F1 + C2*F2)/Fout;

% End time 
t_end = 600;    % [h]

% -------------------------------------------------------------------------
% Solution
% -------------------------------------------------------------------------

opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-12);

% Initial conditions (coming from the ss balances)
IC = Cout;   

[t, C] = ode45(@function_1, [0, t_end], IC, opts,... 
                F2, F1, C1, C2, V);

% -------------------------------------------------------------------------
% Graphical-Post-Processing
% -------------------------------------------------------------------------

figure(1)
plot(t, C, 'b', 'LineWidth', 3)
set(gca, 'Fontsize', 12)
legend('Concentration','Location','best')
xlabel('Time [h]')
ylabel('Concentration of A [K]')
title('Dynamics of CA in a mixer')

% -------------------------------------------------------------------------
% Functions
% -------------------------------------------------------------------------

function yy = function_1(t, C, F2, F1, C1, C2, V)
    
    F1 = F1 + 0.04*t;
    if F1 > 20
        F1 = 20;
    end
    
    Fout = F1 + F2;
    
    yy = (F1*C1 + F2*C2 - Fout*C)/V;
   
end
