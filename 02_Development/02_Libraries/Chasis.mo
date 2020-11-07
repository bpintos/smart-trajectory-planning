block Chasis
  Modelica.Blocks.Interfaces.RealInput delta annotation(
    Placement(visible = true, transformation(extent = {{-100, 10}, {-60, 50}}, rotation = 0), iconTransformation(origin = {-74, 30}, extent = {{-14, -14}, {14, 14}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput Pedal annotation(
    Placement(visible = true, transformation(extent = {{-100, -20}, {-60, 20}}, rotation = 0), iconTransformation(origin = {-74, 0}, extent = {{-14, -14}, {14, 14}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput F_ext annotation(
    Placement(visible = true, transformation(extent = {{-100, -50}, {-60, -10}}, rotation = 0), iconTransformation(origin = {-74, -30}, extent = {{-14, -14}, {14, 14}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput x_out annotation(
    Placement(visible = true, transformation(extent = {{60, 20}, {80, 40}}, rotation = 0), iconTransformation(extent = {{60, 20}, {80, 40}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput y_out annotation(
    Placement(visible = true, transformation(extent = {{60, 0}, {80, 20}}, rotation = 0), iconTransformation(extent = {{60, 0}, {80, 20}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput theta_out annotation(
    Placement(visible = true, transformation(extent = {{60, -20}, {80, 0}}, rotation = 0), iconTransformation(extent = {{60, -20}, {80, 0}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput vel_out annotation(
    Placement(visible = true, transformation(extent = {{60, -40}, {80, -20}}, rotation = 0), iconTransformation(extent = {{60, -40}, {80, -20}}, rotation = 0)));
  constant Real pi = 2 * Modelica.Math.asin(1.0);
  constant Real g = 9.8;
  parameter Real lf = 0.12;
  parameter Real lr = 0.16;
  parameter Real mass = 1;
  parameter Real rho = 1.29;
  parameter Real area_front = 0.025;
  parameter Real coeff_aero = 0.4;
  parameter Real wheel_radius = 0.04;
  parameter Real trans_diff = 1/1.5;
  parameter Real trans = 17/48;
  parameter Real coeff_friction = 1.2;
  parameter Real static_friction = 1;
  parameter Real torque_max = 0.2;
  parameter Real tire_press = 3;
  parameter Real engine_inertia = 0;
  parameter Real trans_inertia = 0;
  parameter Real wheel_inertia = 0;
  Real theta(start = 0);
  Real x(start = 0);
  Real y(start = 0);
  Real vel(start = 0);
  Real F_rolling;
  Real F_aero;
  Real F_friction;
  Real F_res;
  Real F_wheel;
  Real T_motor;
  Real T_trans;
  Real T_wheel;
  Real delta_rad;
  Real beta;
  Real vel_heading;
  Real coeff_rolling;
  Real reduced_inertia;
  Real generalized_mass;
equation
  //Pedal to motor torque
  T_motor = Pedal * torque_max;
  //Motor torque to transmission torque
  T_trans = T_motor/trans;
  //Torque transmission to torque wheel
  T_wheel = T_trans/trans_diff;
  //Torque to force conversion
  F_wheel = T_wheel / wheel_radius;
  //Resistance forces
  F_aero = 0.5 * rho * area_front * coeff_aero * vel ^ 2; //Aerodynamic resistance
  coeff_rolling = 0.005 + (1/tire_press)*(0.01 + 0.0095 * (vel*3.6/100)^2);
  if vel>0 then
    F_rolling = coeff_rolling*mass*g; //Rolling resistance
    F_friction = static_friction + coeff_friction * vel; //Friction resistance
  else
    F_rolling = 0; //Rolling resistance
    F_friction = 0; //Friction resistance
  end if;
  F_res = F_aero + F_rolling + F_friction + F_ext; //Sum of resistances
  //Generalized vehicle mass
  reduced_inertia = wheel_inertia + trans_diff^2 * (trans_inertia + trans^2 * engine_inertia);
  generalized_mass = mass + reduced_inertia/wheel_radius^2;
  //Vehicle longitudinal dynamic
  der(vel) * mass = F_wheel - F_res;
  //Vehicle lateral dynamic
  delta_rad = delta*pi/180;
  beta = atan(lf * tan(delta_rad) / (lf + lr));
  der(theta) = vel * tan(delta_rad) * cos(beta) / (lf + lr);
  vel_heading = mod(beta + delta_rad,2*pi);
  //Vehicle coordinates calculation
  der(x) = vel * cos(theta + beta);
  der(y) = vel * sin(theta + beta);
  //Block output
  x_out = x;
  y_out = y;
  theta_out = mod(theta,2*pi)*180/pi;
  vel_out = vel*3.6;
  annotation(
    Diagram(graphics = {Rectangle(origin = {-0.254174, -0.0745469}, extent = {{-40.0222, 39.6619}, {40.0222, -39.6619}}), Line(origin = {49.6789, 9.63303}, points = {{-10, 0}, {10, 0}}), Rectangle(origin = {0, -8}, extent = {{-26, 10}, {26, -10}}), Rectangle(origin = {-16, -22}, extent = {{-4, 4}, {4, -4}}), Rectangle(origin = {16, -22}, extent = {{4, 4}, {-4, -4}}), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Ellipse(origin = {18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Line(origin = {0.321101, 14.1414}, points = {{-22, -11.8937}, {-16, 12.1063}, {16, 12.1063}, {22, -11.8937}}), Line(origin = {-50.3211, 29.9541}, points = {{-10, 0}, {10, 0}}), Line(origin = {-50.3211, -0.0917431}, points = {{-10, 0}, {10, 0}}), Line(origin = {50, 30}, points = {{10, 0}, {-10, 0}}), Line(origin = {50, -10}, points = {{10, 0}, {-10, 0}}), Line(origin = {-50.3211, 29.9541}, points = {{-10, 0}, {10, 0}}), Line(origin = {-50.3211, -30}, points = {{-10, 0}, {10, 0}}), Line(origin = {50, -10}, points = {{10, 0}, {-10, 0}}), Line(origin = {50, -30}, points = {{10, 0}, {-10, 0}})}),
    Icon(graphics = {Rectangle(origin = {-0.254174, -0.0745469}, extent = {{-40.0222, 39.6619}, {40.0222, -39.6619}}), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Rectangle(origin = {-16, -22}, extent = {{-4, 4}, {4, -4}}), Line(origin = {-50, -0.229358}, points = {{-10, 0}, {10, 0}}), Line(origin = {50, 30}, points = {{10, 0}, {-10, 0}}), Rectangle(origin = {0, -8}, extent = {{-26, 10}, {26, -10}}), Ellipse(origin = {18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Line(origin = {-50.3211, -29.9541}, points = {{-10, 0}, {10, 0}}), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Line(origin = {50, 10}, points = {{-10, 0}, {10, 0}}), Ellipse(origin = {-18, -6}, extent = {{-4, 4}, {4, -4}}, endAngle = 360), Rectangle(origin = {16, -22}, extent = {{4, 4}, {-4, -4}}), Line(origin = {50, -10}, points = {{10, 0}, {-10, 0}}), Line(origin = {0.321101, 14.1414}, points = {{-22, -11.8937}, {-16, 12.1063}, {16, 12.1063}, {22, -11.8937}}), Line(origin = {-50.6422, 30}, points = {{-10, 0}, {10, 0}}), Line(origin = {50, -30}, points = {{10, 0}, {-10, 0}})}));
end Chasis;
