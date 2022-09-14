System-name: default_name
States:
  x:
    Min: -1.0
    Max: 1.0
    X0: 0.0
  y:
    Min: -1.0
    Max: 1.0
    X0: 0.0
  z:
    Min: -1.0
    Max: 1.0
    X0: 0.0
  q0:
    Min: -1.0
    Max: 1.0
    X0: 1.0
  q1:
    Min: -1.0
    Max: 1.0
    X0: 0.0
  q2:
    Min: -1.0
    Max: 1.0
    X0: 0.0
  q3:
    Min: -1.0
    Max: 1.0
    X0: 0.0
  u:
    Min: -1.0
    Max: 1.0
    X0: 20.0
  v:
    Min: -1.0
    Max: 1.0
    X0: 30.0
  w:
    Min: -1.0
    Max: 1.0
    X0: 25.0
  p:
    Min: -1.0
    Max: 1.0
    X0: 1.0
  q:
    Min: -1.0
    Max: 1.0
    X0: 3.0
  r:
    Min: -1.0
    Max: 1.0
    X0: 2.0
Parameters:
  I_13:
    Type: Param
    X0: -0.25
    Min: '-'
    Max: '-'
  mass:
    Type: Param
    X0: 10.0
    Min: '-'
    Max: '-'
  area:
    Type: Param
    X0: 1.0
    Min: '-'
    Max: '-'
  air_density:
    Type: Param
    X0: 1.225
    Min: '-'
    Max: '-'
  I_11:
    Type: Param
    X0: 0.666
    Min: '-'
    Max: '-'
  I_33:
    Type: Param
    X0: 0.666
    Min: '-'
    Max: '-'
  drag_coef:
    Type: Param
    X0: 1.0
    Min: '-'
    Max: '-'
  I_2:
    Type: Param
    X0: 0.666
    Min: '-'
    Max: '-'
  cross-wind:
    Type: Param
    X0: 0.0
    Min: '-'
    Max: '-'
