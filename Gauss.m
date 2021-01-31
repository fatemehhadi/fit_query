function y = Gauss(x, mu, sd, amplitude)
y = amplitude/(2*pi*sd)*exp(-(x-mu).^2/(2*sd^2));