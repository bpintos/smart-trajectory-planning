model Circuit
  Modelica.Blocks.Sources.TimeTable timeTable(table = [0, 0; 2, 0; 4, 2; 6, 2; 8, 0; 10, 0; 12, -5; 15, -5; 17, 0; 20, 0; 23, -1.82; 25, -1.82; 27, 0; 31, 0; 33, -5.05; 36, -5.05; 38, 0; 38.25, 0])  annotation(
    Placement(visible = true, transformation(origin = {-44, 14}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Chasis chasis annotation(
    Placement(visible = true, transformation(origin = {1, 13}, extent = {{-23, -23}, {23, 23}}, rotation = 0)));
equation
  connect(timeTable.y, chasis.delta) annotation(
    Line(points = {{-32, 14}, {-16, 14}, {-16, 14}, {-16, 14}}, color = {0, 0, 127}));
  annotation(
    uses(Modelica(version = "3.2.3")));
end Circuit;
