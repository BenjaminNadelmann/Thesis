
import gurobipy as gp
from gurobipy import GRB

try:
    # Created a new model named "my_milp_model"
    m = gp.Model("my_milp_model")

    # Create binary decision variables
    x = m.addVar(vtype=GRB.CONTINUOUS, name="x")
    y = m.addVar(vtype=GRB.INTEGER, name="y")
    z = m.addVar(vtype=GRB.BINARY, name="z")

    # Defined the objective function (modified for our unique problem)
    m.setObjective(2 * x + 3 * y + z, GRB.MAXIMIZE)

    # Added constraints (modifed coefficients and an additional constraint)
    m.addConstr(2 * x + 3 * y + 5 * z <= 7, "c0")
    m.addConstr(x + 3 * y >= 2, "c1")
    m.addConstr(3 * x - y + z <= 6, "c2")

    # Solved the optimization problem
    m.optimize()

    # Printed the values of the decision variables
    for v in m.getVars():
        print(f"{v.VarName} {v.X:g}")

    # Printed the optimal objective function value
    print(f"Obj: {m.ObjVal:g}")

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")
