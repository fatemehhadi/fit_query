% X-axis (Wavelengths)
wavelength_len = 1000;
wavelength_range =  linspace(100, 200, wavelength_len);

% Four different components
% Component A
mu_a1 = 135;
sigma_a1 = 2;
intensity_a1 = 1;
mu_a2 = 185;
sigma_a2 = 2;
intensity_a2 = 0.4;
gauss_a =  Gauss(wavelength_range, mu_a1, sigma_a1, intensity_a1) + Gauss(wavelength_range, mu_a2, sigma_a2, intensity_a2);

% Component B
mu_b = 150;
sigma_b = 15;
intensity_b = 1;
gauss_b = Gauss(wavelength_range, mu_b, sigma_b, intensity_b);

% Component C
mu_c1 = 120;
sigma_c1 = 2;
intensity_c1 = 0.15;
mu_c2 = 165;
sigma_c2 = 8;
intensity_c2 = 1;
gauss_c = Gauss(wavelength_range, mu_c1, sigma_c1, intensity_c1) + Gauss(wavelength_range, mu_c2, sigma_c2, intensity_c2);

% Component D
mu_d1 = 115;
sigma_d1 = 5;
intensity_d1 = 1;
mu_d2 = 140;
sigma_d2 = 5;
intensity_d2 = 0.85;
gauss_d = Gauss(wavelength_range, mu_d1, sigma_d1, intensity_d1) + Gauss(wavelength_range, mu_d2, sigma_d2, intensity_d2);

% Spectra normalization:
component_a = gauss_a/max(gauss_a);
component_b = gauss_b/max(gauss_b);
component_c = gauss_c/max(gauss_c);
component_d = gauss_d/max(gauss_d);

% Generate the components matrix
components =[component_a; component_b; component_c; component_d];

% Rename the library spectra
X = components;
X = X';
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

%  Setup the data matrix appropriately
[m, n] = size(X);

% What concentrations we want them to have in our query spectrum:
c_a = 0.25;
c_b = 0.7;
c_c = -0.1;
c_d = 0.35;

% Let's build the spectrum to be studied: The query spectrum
query_spectrum = c_a * component_a + c_b * component_b + c_c *component_c + c_d *component_d;

% Let's add it some noise for a bit of realism:
query_spectrum = query_spectrum +   0.02*rand(1, wavelength_len);

% Rename the query spectrum
y = query_spectrum;
y = y';

% Initialize the fitting parameters
initial_theta = zeros(n, 1);

% Plot library spectra
figure
plot(wavelength_range,X(:,1:end))
xlabel('wavenumber')
ylabel('normalized amplitude')
legend('Component 1','Component 2','Component 3','Component 4')

% Analytical solution
theta_nomral_equation = inv(X'*X)*X'*y;
h_nomral_equation=X*theta_nomral_equation;

% Set gradient descent parameters
lambda = 0;
iterations = 100;
alpha = 0.5;

% Run gradient descent:
[theta, J_history] = gradientDescent_GitHub(X, y, initial_theta, alpha, iterations, lambda);

figure
plot(1:1:iterations,J_history)
xlabel('iteration')
ylabel('objective function')

% %  Set options for fminunc
% lambda = 0;
% options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 10000);
% 
% %  Run fminunc to obtain the optimal theta
% %  This function will return theta and the cost 
% [theta_fminunc, J_fminunc] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
% csvwrite('theta_fminunc.csv', theta_fminunc)
% 
% %  Set options for fmincon
% lb = zeros(n+1, 1);
% lb = lb';
% ub = zeros(n+1, 1);
% ub = ub';
% ub(1,:) = 2;
% 
% A = [];
% b = [];
% Aeq = [];
% beq = [];
% 
% %  Run fmincon to obtain the optimal theta
% %  This function will return theta and the cost
% initial_theta(3,1) = 1;
% [theta_fmincon, J_fmincon] = fmincon(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta,A,b,Aeq,beq,lb,ub);
% csvwrite('theta_fmincon.csv', theta_fmincon)

% Plot results
figure
plot(wavelength_range,y,'r')

hold on
plot(wavelength_range,h_nomral_equation,'g')

hold on
h=X*theta;
plot(wavelength_range,h,'b')

% hold on
% h_fminunc=X*theta_fminunc;
% plot(wavelength_range,h_fminunc,'m')
% 
% hold on
% h_fmincon=X*theta_fmincon;
% plot(wavelength_range,h_fmincon,'c')

xlabel('wavenumber')
ylabel('normalized amplitude')
legend('query','normal equation','gradient descent')
%legend('query','normal equation','gradient descent','fminunc','fmincon','starphire')

% Compare objective functions
J_normal = sum((h_nomral_equation-y).^2)/(2*m)
J = sum((h-y).^2)/(2*m)

% %J3 = sum((h_fminunc-y).^2)/(2*m)
% J_fminunc
% 
% %J4 = sum((h_fmincon-y).^2)/(2*m)
% J_fmincon