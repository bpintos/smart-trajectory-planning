block Chasis
  Modelica.Blocks.Interfaces.RealInput delta annotation(
    Placement(visible = true, transformation(extent = {{-100, -20}, {-60, 20}}, rotation = 0), iconTransformation(origin = {-74, 0}, extent = {{-14, -14}, {14, 14}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput x_out annotation(
    Placement(visible = true, transformation(extent = {{60, 10}, {80, 30}}, rotation = 0), iconTransformation(extent = {{60, 10}, {80, 30}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput y_out annotation(
    Placement(visible = true, transformation(extent = {{60, -10}, {80, 10}}, rotation = 0), iconTransformation(extent = {{60, -10}, {80, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput theta_out annotation(
    Placement(visible = true, transformation(extent = {{60, -30}, {80, -10}}, rotation = 0), iconTransformation(extent = {{60, -30}, {80, -10}}, rotation = 0)));
  constant Real pi = 2 * Modelica.Math.asin(1.0);
  constant Real vel = 2;
  parameter Real lf = 0.12;
  parameter Real lr = 0.16;
  Real theta(start = 0*pi/180);
  Real x(start = 0);
  Real y(start = 0);
  Real delta_rad;
  Real beta;
  Real vel_heading;
equation
//Vehicle lateral dynamic
  delta_rad = delta * pi / 180;
  beta = atan(lr * tan(delta_rad) / (lf + lr));
  der(theta) = vel * tan(delta_rad) * cos(beta) / (lf + lr);
  vel_heading = mod(beta + delta_rad,2*pi);
//Vehicle coordinates calculation
  der(x) = vel * cos(theta + beta);
  der(y) = vel * sin(theta + beta);
//Block output
  x_out = x;
  y_out = y;
  theta_out = mod(theta,2*pi)*180/pi;
  annotation(
    Diagram(graphics = {Rectangle(origin = {-0.254174, -0.0745469}, extent = {{-40.0222, 39.6619}, {40.0222, -39.6619}}), Line(origin = {49.6789, -0.36697}, points = {{-10, 0}, {10, 0}}), Rectangle(origin = {0, -8}, extent = {{-26, 10}, {26, -10}}), Rectangle(origin = {-16, -22}, extent = {{-4, 4}, {4, -4}}), Rectangle(origin = {16, -22}, extent = {{4, 4}, {-4, -4}}), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Ellipse(origin = {18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Line(origin = {0.321101, 14.1414}, points = {{-22, -11.8937}, {-16, 12.1063}, {16, 12.1063}, {22, -11.8937}}), Line(origin = {-50.3211, -0.0459001}, points = {{-10, 0}, {10, 0}}), Line(origin = {50, 20}, points = {{10, 0}, {-10, 0}}), Line(origin = {50, -20}, points = {{10, 0}, {-10, 0}}), Line(origin = {-50.3211, -0.0459001}, points = {{-10, 0}, {10, 0}}), Line(origin = {50, -20}, points = {{10, 0}, {-10, 0}})}),
    Icon(graphics = {Rectangle(origin = {-0.254174, -0.0745469}, extent = {{-40.0222, 39.6619}, {40.0222, -39.6619}}), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Rectangle(origin = {-16, -22}, extent = {{-4, 4}, {4, -4}}), Line(origin = {50, 20}, points = {{10, 0}, {-10, 0}}), Rectangle(origin = {0, -8}, extent = {{-26, 10}, {26, -10}}), Ellipse(origin = {18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Line(origin = {50, 0}, points = {{-10, 0}, {10, 0}}), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Rectangle(origin = {16, -22}, extent = {{4, 4}, {-4, -4}}), Line(origin = {50, -20}, points = {{10, 0}, {-10, 0}}), Line(origin = {0.321101, 14.1414}, points = {{-22, -11.8937}, {-16, 12.1063}, {16, 12.1063}, {22, -11.8937}}), Line(origin = {-50.6422, 0}, points = {{-10, 0}, {10, 0}})}));
end Chasis;
