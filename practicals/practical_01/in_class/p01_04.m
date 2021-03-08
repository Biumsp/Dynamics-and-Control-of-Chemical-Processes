% Enrico Bussetti
% Dynamics and Control of Chemical Processes
% Practical 1, exercise 4

close all
clear variables

format compact
format short g

% -------------------------------------------------------------------------
% Data
% -------------------------------------------------------------------------

Tc = 300;       % Coolant Temperature [K]
UA = 150;       % [J/min/K]

DHr = -5e3;     % [J/mol]
A   = 600;      % [1/min]
Ea  = 3e4;      % [J/mol]
R   = 8.314;    % [J%mol/K]

cp  = 25;       % [J/mol/K]
rho = 1200;     % [kg/m^3]

CA0 = 0.03;     % [mol/m^3]
CB0 = 0;        % [mol/m^3]

Tin = 300;      % [K]
Fin = 0.005;    % [m^3/min]
V0  = 0.785;    % [m^3]
F   = 0.005;    % [m^3/min]

% End time 
t_end = 300;    % [s]

% -------------------------------------------------------------------------
% Solution
% -------------------------------------------------------------------------

opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-12);

% Initial conditions
IC = [V0, CA0, Tin, CB0];
[t, var] = ode15s(@function_1, [0, t_end], IC, opts, ...
                  Fin, F, CA0, rho, cp, Tin, UA, A, Ea, R, Tc, DHr);
            
V = var(:, 1);
CA = var(:, 2);
T = var(:, 3);
CB = var(:, 4);
    
% -------------------------------------------------------------------------
% Graphical-Post-Processing
% -------------------------------------------------------------------------

figure(1)
plot(t, T, 'b', 'LineWidth', 2)
set(gca, 'Fontsize', 14)
legend('Temperature','Location','best')
xlabel('Time [min]')
ylabel('Temperature [K]')
title('Temperature of CSTR')

figure(2)
plot(t, CA, 'r', 'LineWidth', 2)
set(gca, 'Fontsize', 14)
legend('Concentration of A','Location','best')
xlabel('Time [min]')
ylabel('Concentration of A [mol/m^3]')
title('Concentration of A')

figure(3)
plot(t, V, 'b', 'LineWidth', 2)
set(gca, 'Fontsize', 14)
legend('Volume','Location','best')
xlabel('Time [min]')
ylabel('Volume [m^3]')
title('Volume')

figure(4)
plot(t, CB, 'r', 'LineWidth', 2)
set(gca, 'Fontsize', 14)
legend('Concentration of B','Location','best')
xlabel('Time [min]')
ylabel('Concentration of B [mol/m^3]')
title('Concentration of B')

% -------------------------------------------------------------------------
% Functions
% -------------------------------------------------------------------------

function yy = function_1(t, var, Fin, F, CA0, rho, cp, Tin, UA, A, Ea, R, Tc, DHr)
    
    V = var(1);
    CA = var(2);
    T = var(3);
    CB = var(4);
    
    yy = zeros(4, 1);
    
    r = A*exp(-Ea/R/T)*CA;
        
    yy(1) =  Fin - F;
    yy(2) =  Fin/V*(CA0 - CA) - r;
    yy(3) = -UA*(T - Tc)/rho/cp/V + Fin/V*(Tin - T) - DHr*r/rho/cp;
    yy(4) = -CB*F/V + r;
    
end
