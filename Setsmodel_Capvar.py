# RES neutral emission model

import numpy as np
from gurobipy import GRB
import gurobipy as gp
import random
import matplotlib.pyplot as plt

T = 5 #Number of Timesteps
W = 10 #Number of Scenarios
G = 5 #Number of Generators
GC = [1,1,1,0,0] #Generator placement (1 for conventional, 0 for renewable)
GR = [0,0,0,1,1] #Generator placement (0 for conventional, 1 for renewable)
#RCP = [1,1,1,0.6,0.6] #Capacity factors for the generators, thought it could be used as a range
total_time = 8760  # Total sum required for each scenario
time_lb = 900
time_ub = 2750
random.seed(201)

pi_w = {i: 1/W for i in range(W)}

tau_tw = {}

# Populate the nested dictionary with values
for w in range(W):
    remaining_sum = total_time
    tau_timeval = []

    # Distribute the total sum randomly among T timesteps
    for t in range(T - 1):
        value = random.randint(time_lb, min(time_ub, remaining_sum - (T - t - 1) * time_lb))
        tau_timeval.append(value)
        remaining_sum -= value

    # The last value ensures the total sum constraint
    tau_timeval.append(remaining_sum)

    # Shuffle the values to make it more random
    random.shuffle(tau_timeval)

    for t in range(T):
        key = (t, w)
        value = tau_timeval[t]  # Extract the value corresponding to the timestep
        tau_tw[key] = value

#print(tau_tw)

#Beta_values = [50, 75, 100, 125, 150]
Beta_values = [50, 61, 72, 83, 94, 106, 117, 128, 139, 150]

# Initialize an empty nested dictionary
beta_tw = {}

# Populate the nested dictionary with values
for t in range(T):
    for w in range(W):
        key = (t, w)
        value = Beta_values[w]  # Extract the value corresponding to the timestep
        beta_tw[key] = value

#print(beta_tw)

c_g = {g : 30 if GC[g] else 0 for g in range(G)}
i_g = {g : 25000 if GC[g] else 50000 for g in range(G)}
alpha_t = {i: 0.2+0.25*i for i in range(T)}
rho_gtw = {(g, t, w): 0.75 if GC[g] else 0.25
           for g in range(G) for t in range(T) for w in range(W)}

#print(rho_gtw)
q_g0 = {g : 60 if GC[g] else 30 for g in range(G)} #Initial capacity of conventional = 1000 MW, renewable = 2300 MW
kappa_w = {i: 0.3 for i in range(W)}
Phi = -1
conjectural_t = {i: -alpha_t[i]*(1+Phi) for i in range(T)} #Not sure if these parameters are actually parameters

#e_gcap = {g : 300 if GC[g] else 0 for g in range(G)}
P_twCO2 = {(t, w): 200 for t in range(T) for w in range(W)}
P_gCO2 = {g : 0 if GC[g] else 0 for g in range(G)}
Zeta_gtw = {(g, t, w): 1 if GC[g] else 0
           for g in range(G) for t in range(T) for w in range(W)}


def MIQP_RES_Emis(pi_w, tau_tw, beta_tw, alpha_t, c_g, i_g, rho_gtw, q_g0, kappa_w, conjectural_t, T, W, G, P_twCO2, P_gCO2, Zeta_gtw):

    m=gp.Model('RES-model')

    #Defining the primal variables:
    q_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Quantity")
    q_gbar=m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name="New_Capacity")
    d_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Demand")
    p_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Price")
    eps_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Emission")
    e_g=m.addVars(G, lb=0, ub=10000, vtype=GRB.CONTINUOUS, name="Emission permit")

    #Defining the Dual Variables:
    eta_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-1")
    psi_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-2")
    phi_g=m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name="Dual-3")
    l_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-4")
    xi_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-5")

    #Defining the Binary variables:
    b1 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable1")
    b2 = m.addVars(G, vtype=GRB.BINARY, name="Binary_Variable2")
    b3 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable3")
    b4 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable4")

    #Defining the Big M:
    M = 50000
    M1 = 1000

    #Defining the Primary Constraints
    for t in range(T):
        for w in range(W):
            m.addConstr(gp.quicksum(q_gtw[g,t,w] for g in range(G)) == d_tw[t,w] , name="1.8a")
            m.addConstr(p_tw[t,w] == beta_tw[t,w] - alpha_t[t] * d_tw[t,w], name="1.8c")

            for g in range(G):
                m.addConstr(eps_gtw[g,t,w] == Zeta_gtw[g,t,w] * q_gtw[g,t,w] - e_g[g], name="emission")

                m.addConstr(eps_gtw[g,t,w] <= M * b4[g,t,w], name="BigM8")
                m.addConstr(l_gtw[g,t,w] <= M * (1-b4[g,t,w]), name="BigM9")

                m.addConstr(l_gtw[g,t,w] - P_twCO2[t,w] * pi_w[w] * tau_tw[t,w] + xi_tw[t,w] == 0, name="1.8d")
                # I might need to add the partial derivative of the e_g as well just not sure.

    # for g in range(G):
    #     m.addConstr(e_g[g] == gp.quicksum(pi_w[w] * tau_tw[t, w] * (Zeta_gtw[g, t, w] * q_gtw[g,t,w]) - eps_gtw[g,t,w] for t in range(T) for w in range(W)), name = 'E_perm') 


    for w in range(W):
        m.addConstr(gp.quicksum(tau_tw[t,w] * gp.quicksum(q_gtw[g,t,w] for g in (3,4)) for t in range(T)) #Only renewable generators should be considered global penetration
                    >= kappa_w[w] * gp.quicksum(tau_tw[t,w] * d_tw[t,w] for t in range(T)), name="1.8b")

    #Defining the complementarity constraints
    for g in range(G):
        for t in range(T):
            for w in range(W):
                
                m.addConstr(-pi_w[w] * tau_tw[t,w] * (p_tw[t,w] + conjectural_t[t] * q_gtw[g,t,w] - c_g[g])
                             + eta_gtw[g,t,w] + Zeta_gtw[g,t,w] * xi_tw[t,w] - psi_gtw[g,t,w] == 0, name="Q_gtw")

                m.addConstr(q_gtw[g,t,w] <= M * b1[g,t,w], name="BigM1")
                m.addConstr(psi_gtw[g,t,w] <= M * (1-b1[g,t,w]), name="BigM2")

                m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] >= 0, name="BigM5")
                m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] <= M * (1-b3[g,t,w]), name="BigM6")
                m.addConstr(eta_gtw[g,t,w] <= M * b3[g,t,w], name="BigM7")

        #for variables and parameters with purely g
        m.addConstr(i_g[g] - gp.quicksum(rho_gtw[g,t,w] * eta_gtw[g,t,w] for t in range(T) for w in range(W)) - phi_g[g] == 0, name="Q_gbar")

        m.addConstr(q_gbar[g] <= M * b2[g], name="BigM3")
        m.addConstr(phi_g[g] <= M * (1-b2[g]), name="BigM4")
 
    #Writing out the objective function - Only conventional generators that should be considered for emissions               
    objective = gp.quicksum( pi_w[w] * tau_tw[t, w] * (beta_tw[t, w] * d_tw[t, w] - (1/2) * alpha_t[t] * d_tw[t, w] * d_tw[t, w] 
    - gp.quicksum(c_g[g] * q_gtw[g, t, w] - P_twCO2[t,w] *(e_g[g] - (Zeta_gtw[g, t, w] * q_gtw[g,t,w])) for g in range(G))) for w in range(W) for t in range(T))
    - gp.quicksum(i_g[g] * q_gbar[g] + P_gCO2[g] * e_g[g] for g in range(G))
    

# Set objective function
    m.setObjective(objective, GRB.MAXIMIZE)
 
    m.update()
    m.optimize()
 
    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        # Print out the IIS constraints and variables
        print('\nThe following constraints and variables are in the IIS:')
        for c in m.getConstrs():
            if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')
 
        for v in m.getVars():
            if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
            if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
 
    if m.Status == GRB.OPTIMAL:
        m.printAttr('X')
        Con = np.zeros(W)
 
        # Iterate through scenarios
        for scenario in range(W):
            # Initialize the sum for this scenario
            scenario_sum = 0
 
            # Iterate through generators and time values
            for g in (0,1,2):
                for t in range(T):
                    # Replace with your actual Gurobi variable indexing
                    # Assuming q_gtw[g, t, w].X gives the value for each w
                    scenario_sum += q_gtw[g, t, scenario].X  # Corrected indexing here
 
            # Store the scenario sum in the array
            Con[scenario] = scenario_sum
       
        Ren = np.zeros(W)
 
        # Iterate through scenarios
        for scenario in range(W):
            # Initialize the sum for this scenario
            scenario_sum = 0
 
            # Iterate through generators and time values
            for g in (3,4):
                for t in range(T):
                    # Replace with your actual Gurobi variable indexing
                    # Assuming q_gtw[g, t, w].X gives the value for each w
                    scenario_sum += q_gtw[g, t, scenario].X  # Corrected indexing here
 
            # Store the scenario sum in the array
            Ren[scenario] = scenario_sum

        emission = np.zeros(3)
 
        # Iterate through scenarios
        for Gen in (0,1,2):
            # Initialize the sum for this scenario
            Ems_sum = 0
 
            # Iterate through generators and time values
            for t in range(T):
                for w in range(W):
                    # Replace with your actual Gurobi variable indexing
                    # Assuming q_gtw[g, t, w].X gives the value for each w
                    Ems_sum += eps_gtw[Gen, t, w].X  # Corrected indexing here
 
            # Store the scenario sum in the array
            emission[Gen] = Ems_sum

        New_capacity = np.zeros(G)
        for g in range(G):
            New_capacity[g] = q_gbar[g].X
        
        Emission_permits = np.zeros(G)
        for g in range(G):
            Emission_permits[g] = e_g[g].X
        
        obj = m.getObjective()
        print(obj.getValue() * 10 ** -6, "mEuro")
 
    return Con, Ren, New_capacity, emission, Emission_permits

Quantity, Quantity2, New_capacity, emission, E_permits = MIQP_RES_Emis(pi_w, tau_tw, beta_tw, alpha_t, c_g, i_g, rho_gtw,
                                                             q_g0, kappa_w, conjectural_t, T, W, G, P_twCO2, P_gCO2, Zeta_gtw)

print("Conventional quantity generated per scenario", Quantity)
print("Renewable quantity generated per scenario", Quantity2)
print("Emission per generator", emission)
print("percentage of renewable quantity", np.sum(Quantity2)/(np.sum(Quantity)+np.sum(Quantity2))*100, "%")
#print("Emission profit / cost ", (e_g- emission) * P_twCO2[0,0] )
print("New capacity", New_capacity)
print("Bought emission permits", E_permits)


S = np.arange(0,10)
Generators = np.arange(0,5)
Quantities = {'Ren': Quantity2,'Conv': Quantity,}

fig, axs = plt.subplots(2)
bottom = np.zeros(10)

bar_labels = ('S0', 'S1', 'S2', 'S3','S4', 'S5','S6', 'S7','S8', 'S9')

for boolean, weight_count in Quantities.items():
    p = axs[0].bar(bar_labels, weight_count, label=boolean, bottom=bottom)
    bottom += weight_count

axs[0].set_title("Total Quantity of Energy Produced by Generators")
axs[0].legend(loc="upper left")

bar_l1 = ['Con', '_Con', '_Con', 'Ren', '_Ren']
bar_c1 = ['tab:red', 'tab:red', 'tab:red', 'tab:blue', 'tab:blue']

axs[1].bar(Generators, New_capacity, label=bar_l1, color=bar_c1)

axs[1].set_ylabel('New capacity')
axs[1].legend(title='Generator type', loc="upper left")

plt.show()