# Dynamic system

A general container for dynamic system with useful methods. System variables and
functions use the `SymPy` package. 

A continuous-time dynamic system is described by
$$ \boldsymbol{x'} _k = f( \boldsymbol{x} _{k-1}, \boldsymbol{u}(t),
\boldsymbol{P} ) $$

The Dynamic system is contained in the separate 'dynamicsystem' package. 
It integrates the continuous-time dynamic equations to get the model-estimated state. 

Parameters $\boldsymbol{P}$ and initial values $\boldsymbol{x}_0$ are read from
a config file.
 
The main class methods include:

### `write_default_config` 
Write a config file with parameter and initial state values.

### `load_config`
Load the parameter values from the config file.

### `load_x_0` 
Load the initial state from the config file

### `add_aug_states`
Parameters can be treated as augmented states. This can be useful when
linearising systems. This method reads the config file and adjusts states as
necessary.

### `sub_params`
Substitute parameter values into the symbolic dynamic equations

### `create_jacobian`
As the name suggests, this creates a Jacobian function $J(\boldsymbol{x})$ which
calculates the gradient matrix. 

### `lambdify`
This creates `NumPy` expressions of the dynamic functions which take vector
inputs. Dramatic speed improvements for evaluating functions (such as for
time-step integration).

### `get_x_dot`
Returns $\frac{dx}{dt}$ using the lambdified dynamic equations.

### `step`
An integration time-step using an RK4 integrator.

### `integrate`
Integrates the dynamic functions over some time interval. A minimum timestep is
specified, and `step` is called for each timestep over the total time interval. 

### `save` 
Saves the dynamic system as as pickle package

