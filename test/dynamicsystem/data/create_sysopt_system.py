from sysopt_symbolic import Block, Metadata, Composite
from sysopt_symbolic.solver import SymbolDatabase
from sysopt_symbolic.backends import simplify_system
from dynamicsystem import DynamicSystem

class Position(Block):
    def __init__(self):
        metadata = Metadata(
            states = ["x",
                     "y",
                     "z"],
            inputs=["u",
                    "v",
                    "w"],
            outputs=["x",
                     "y",
                     "z"],
            parameters=[]
        )
        super().__init__(metadata)

    def initial_state(self, parameters):
        return [
            0,
            0,
            0,
        ]

    def compute_dynamics(self, t, state, algebraics, inputs, parameters):
        u, v, w = inputs

        return [
            u, 
            v, 
            w
        ]

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        x, y, z = state

        return [
            x, 
            y, 
            z
        ]

class Newton(Block):
    def __init__(self):
        metadata = Metadata(
            states = ["u",
                     "v",
                     "w"],
            inputs=["f1",
                    "f2",
                    "f3"],
            outputs=["u",
                     "v",
                     "w"],
            parameters=["m"]
        )
        super().__init__(metadata)

    def initial_state(self, parameters):
        return [
            0,
            0,
            0
        ]

    def compute_dynamics(self, t, state, algebraics, inputs, parameters):
        u, v, w = state
        f1, f2, f3 = inputs
        m, = parameters

        return [
            f1/m,
            f2/m,
            f3/m
        ]

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        u, v, w = state

        return [
            u,
            v,
            w
        ]

class DoF3(Composite):
    def __init__(self):
        self.position = Position()
        self.Newton = Newton()

        components = [self.position, self.Newton]

        wires = [
            (self.Newton.outputs[0:3], self.position.inputs[0:3]),
        ]
        super().__init__(components, wires)

model = DoF3()
backend = SymbolDatabase()
X, P, f, sym_dict = simplify_system(backend,model)

dsys = DynamicSystem(X,P,f,sym_dict)
dsys.save("DoF3")
