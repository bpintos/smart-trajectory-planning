model VehicleModel
  Modelica.Blocks.Sources.Constant const1(k = 0.5)  annotation(
    Placement(visible = true, transformation(origin = {-78, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant const2(k = 0)  annotation(
    Placement(visible = true, transformation(origin = {-78, -32}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Nonlinear.SlewRateLimiter slewRateLimiter( Rising = 1, y_start = 0)  annotation(
    Placement(visible = true, transformation(origin = {-48, 32}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Nonlinear.Limiter limiter(limitsAtInit = true, uMax = 15)  annotation(
    Placement(visible = true, transformation(origin = {-18, 32}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant const(k = 2)  annotation(
    Placement(visible = true, transformation(origin = {-78, 32}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Chasis chasis annotation(
    Placement(visible = true, transformation(origin = {54, 2}, extent = {{-50, -50}, {50, 50}}, rotation = 0)));
equation
  connect(slewRateLimiter.y, limiter.y) annotation(
    Line(points = {{-36, 32}, {-8, 32}, {-8, 32}, {-6, 32}, {-6, 32}}, color = {0, 0, 127}));
  connect(const.y, slewRateLimiter.u) annotation(
    Line(points = {{-66, 32}, {-62, 32}, {-62, 32}, {-60, 32}}, color = {0, 0, 127}));
  connect(limiter.y, chasis.delta) annotation(
    Line(points = {{-6, 32}, {2, 32}, {2, 18}, {18, 18}, {18, 18}}, color = {0, 0, 127}));
  connect(const1.y, chasis.Pedal) annotation(
    Line(points = {{-66, 0}, {12, 0}, {12, 2}, {18, 2}}, color = {0, 0, 127}));
  connect(const2.y, chasis.F_ext) annotation(
    Line(points = {{-66, -32}, {2, -32}, {2, -14}, {18, -14}, {18, -12}}, color = {0, 0, 127}));
protected
  annotation(
    uses(Modelica(version = "3.2.3")));
end VehicleModel;
