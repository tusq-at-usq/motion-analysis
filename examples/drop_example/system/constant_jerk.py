
""" Full 6-DoF system
"""

import sympy as sp

from sysopt_symbolic import Block, Metadata
from sysopt_symbolic.solver import SymbolDatabase
from sysopt_symbolic.backends import simplify_system
from dynamicsystem.system_class import DynamicSystem


class Physics6DoF(Block):
    """
    6 Degree of Freedom (6-DoF) system:
    - Position is in local coordinates
    - Velocity and acceleration is in body coordiantes
    - Attitude is body frame relative to local fram
    - Rate of rotation and rotational acceleration are in local frame, about
        the cube origin
    - Gravity is hard-coded. No other forces are included
    """
    def __init__(self):
        metadata = Metadata(
            states = ["x",
                      "y",
                      "z",
                      "q0",
                      "q1",
                      "q2",
                      "q3",
                      "v_x",
                      "v_y",
                      "v_z",
                      "p",
                      "q",
                      "r",
                      "a_x",
                      "a_y",
                      "a_z",
                      "j_x",
                      "j_y",
                      "j_z"],
            inputs=[],
            outputs=[],
            parameters=["mass",
                        "I_11",
                        "I_2",
                        "I_33",
                        "I_13"]
        )
        super().__init__(metadata)

    def compute_dynamics(self, t, states, algebraics, inputs, parameters):
        _, _, _, q0, q1, q2, q3, v_x ,v_y, v_z, p, q, r, a_x, a_y, a_z, j_x, j_y, j_k  = states
        mass, I_11, I_2, I_33, I_13 = parameters

        # From Zipfel eq. 10.18
        #TODO: Check these - are they correct?
        p_dot = q/(I_11*I_33 - I_13**2) * ( (I_2*I_33- I_33**2 - I_13**2)*r
                                            - I_13*(I_33+I_11-I_2)*p)

        q_dot = 1./I_2 * ((I_33-I_11)*p*r + I_13*(p**2 - r**2))

        r_dot = q/(I_11*I_33 - I_13**2) * ((-I_11*I_2 + I_11**2 + I_13**2)*p
                                          + I_13*(I_33+I_11-I_2)*r)


        lambd = 1 - (q0**2 + q1**2 + q2**2 + q3**2)
        k = 0.5
        correction = sp.Matrix([[lambd*k], [lambd*k], [lambd*k], [lambd*k]])
        Q = sp.Matrix([[q0], [q1], [q2], [q3]])
        dq_matrix = 0.5*sp.Matrix([
            [0, -p, -q, -r],
            [p, 0,  r,  -q],
            [q, -r, 0,  p],
            [r, q,  -p, 0]
        ])
        #  dqdt = dq_matrix*Q+correction
        dqdt = dq_matrix*Q

        return [
            v_x,
            v_y,
            v_z,
            dqdt[0],
            dqdt[1],
            dqdt[2],
            dqdt[3],
            a_x,
            a_y,
            a_z,
            p_dot,
            q_dot,
            r_dot,
            j_x,
            j_y,
            j_k,
            0,
            0,
            0
        ]

    def compute_outputs(self, t, states, algebraics, inputs, parameters):

        return [
        ]

model = Physics6DoF()
backend = SymbolDatabase()

X, P, f, sym_dict = simplify_system(backend,model)
dsys = DynamicSystem(X,P,f,sym_dict)
dsys.save("constant_jerk")
