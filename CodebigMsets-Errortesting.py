
# import numpy as np
# from gurobipy import GRB
# import gurobipy as gp
# import random

# T = 5 #Number of timesteps
# W = 5 #Number of scenarios
# GC = 3 #Number of Conventional Generators 
# G = 5 #Number of generators

# pi_w = {i: 1/W for i in range(W)}
# #print(pi_w)
# tau_t = {i: 1 for i in range(T)} #should be tau_tw
# #print(tau_tw)
# v_g = {i: 1/G for i in range(G)}
# #print(v_g)
# beta_w = [50, 75, 100, 125, 150]
# #print(beta_tw)
# c_g = [10, 10, 10, 0, 0]
# i_g = [50000, 50000, 50000, 100000, 100000]
# alpha_t = {i: 1/T for i in range(T)}
# zeta = 0.2
# rho = 0.6
# q_g0 = [10, 10, 10, 10, 10] 
# kappa = 0.4 
# P_CO2 = 55 
# ecap = 100 
# conjectural_variation = 1 #Not sure if these parameters are actually parameters
# #beta = 1 #Not sure if these parameters are actually parameters


# def MIQP_Sets(pi_w, tau_t, v_g, beta_w, alpha_t, c_g, i_g, zeta, rho, q_g0, kappa, P_CO2, ecap, conjectural_variation):

#     m=gp.Model('Social_Welfare_Model')
    
#     #Defining the primal variables:
#     q_gtw=m.addVars(G, T, W, lb=0, ub=10000, vtype=GRB.CONTINUOUS, name="quantity")
#     q_gbar=m.addVars(G, lb=0, ub=10000, vtype=GRB.CONTINUOUS, name="New_Capacity")
#     epsilon_gtw=m.addVars(G, T, W, lb=0, ub=10000, vtype=GRB.CONTINUOUS, name="emission") 
#     d_tw=m.addVars(T, W, lb=0, ub=10000, vtype=GRB.CONTINUOUS, name="demand")
#     R_gtw=m.addVars(GC, T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="revenue")
#     C_gtw=m.addVars(GC, T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="cost")
#     sigma_g=m.addVars(G, ub=10000,vtype=GRB.BINARY, name="Auxiliary_Variable1")
#     psi_gw=m.addVars(G, W, lb=0, ub=10000,vtype=GRB.BINARY, name="Auxiliary_Variable2")
#     beta_w = m.addVars(W, lb=0, ub=10000,vtype=GRB.BINARY, name="Auxiliary_Variable3")

#     #Defining the Dual Variables:
#     p_tw=m.addVars(T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-1")
#     eta_gtw=m.addVars(G, T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-2")
#     #underline_gamma_gtw=m.addVars(G, T, W, lb=0,vtype=GRB.CONTINUOUS, name="Dual-3") 
#     #Bar_gamma_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-4")
#     phi_g=m.addVars(G, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-5")
#     delta_gtw=m.addVars(G, T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-6")
#     Theta_gw=m.addVars(G, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-7")
#     muC_gtw=m.addVars(GC, T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-8")
#     muR_gtw=m.addVars(GC, T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-9")
#     l_gtw=m.addVars(G, T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-10")
#     psi_gtw=m.addVars(G, T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-11")
#     xi_tw=m.addVars(T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-12")
#     vR_gtw=m.addVars(G, T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-13")
#     vC_gtw=m.addVars(G, T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-14")
    

#     #The generators' profit optimization model (Maybe it should also be indexed through the different sets)
#     for g in range(G):
#         for gc in range(GC):
#             for t in range(T): 
#                 for w in range(W):
#                     Z_gtw = pi_w[w] * tau_t[t] * (p_tw[t,w] * q_gtw[g,t,w] - (c_g[g] * q_gtw[g,t,w] - P_CO2*(R_gtw[gc,t,w] - C_gtw[gc,t,w])))-i_g[g] * q_gbar[g]

#     #Defining constraints Upper Level:

#     for g in range(G):
#         for t in range(T): 
#             for w in range(W):
#                 m.addConstr(q_gtw[g,t,w] == d_tw[t,w], name="UL-1")
#                 m.addConstr(tau_t[t] * q_gtw[g,t,w] >= kappa * tau_t[t] * d_tw[t,w], name="UL-2") # should be for gr
#                 m.addConstr(p_tw[t,w] == beta_w[w] - alpha_t[t]*d_tw[t,w], name="UL-3")

#     #Defining constraints Lower Level: Equality 

#     for g in range(G):
#         for gc in range(GC):
#             for t in range(T): 
#                 for w in range(W):
#                     m.addConstr(((1 - beta_w[w]-delta_gtw[g,t,w]) * pi_w[w] * tau_t[t] * (p_tw[t,w] + (-alpha_t[t] * (1+conjectural_variation) * q_gtw[g,t,w]) - c_g[g])
#                         + eta_gtw[g,t,w] + zeta * xi_tw[t,w] - psi_gtw[g,t,w] == 0),name="1.7a")
#                     m.addConstr((-eta_gtw[g,t,w] * rho - i_g[g] * (1 - beta_w[w] - delta_gtw[g,t,w]) - phi_g[g] == 0), name="1.7b")
#                     m.addConstr(vC_gtw[g,t,w] - vR_gtw[g,t,w] - xi_tw[t,w] - l_gtw[g,t,w] == 0, name="1.7c")
#                     m.addConstr((1 - beta_w[w] - delta_gtw[g,t,w]) * pi_w[w] * tau_t[t] * P_CO2 - muR_gtw[gc,t,w] - vR_gtw[g,t,w] ==0, name="1.7d")
#                     m.addConstr( (1 - beta_w[w] - delta_gtw[g,t,w]) * pi_w[w] * tau_t[t] * P_CO2 - muC_gtw[gc,t,w] - vC_gtw[g,t,w] == 0, name="1.7e") #In overleaf we have a negative sign infront
#                     m.addConstr( (beta_w[w] * pi_w[w]) / (1 - v_g[g]) - delta_gtw[g,t,w] - Theta_gw[g,w] == 0, name="1.7f") #In overleaf we have a negative sign infront
#                     #m.addConstr( beta_w[w] - delta_gtw[g,t,w] == 0, name="1.7g")

    ## Constraints that only target the conventional generators
    # for gc in range(GC):
    #     for t in range(T): 
    #         for w in range(W):
    #             m.addConstr(epsilon_gtw[gc,t,w] - zeta * q_gtw[gc,t,w] == 0, name="1.7h")

    #             z1 = m.addVar(vtype=GRB.BINARY, name="z1")
    #             z2 = m.addVar(vtype=GRB.BINARY, name="z2")

    #             # Add activation constraints based on the condition
    #             M = 300  # Big M
    #             m.addConstr(R_gtw[gc,t,w] - (ecap - epsilon_gtw[gc,t,w]) <= M * z1, name="Activation_Constraint1")
    #             m.addConstr(C_gtw[gc,t,w] - (epsilon_gtw[gc,t,w] - ecap) <= M * z2, name="Activation_Constraint2")

    #             # Ensure only one activation constraint is active
    #             m.addConstr(z1 + z2 == 1, name="Activation_Constraint")

#     #Defining constraints Lower Level: complementarity 

#     #This is done by linearizing the complementarity constraints through the Big M technique:
    
    # C = 8 #Number of complementarity constraints 
    # b = m.addVars(C, vtype=GRB.BINARY, name="BigM_AV") #Need to set up 8 different binary variables with their given sets
    # M1, M2 = 600, 1000

    # #Big M constraints where all generators are involved 
    # for g in range(G):
    #     for t in range(T): 
    #         for w in range(W):
    #             m.addConstr(rho * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] >= 0, name="P1_BigM1")
    #             m.addConstr(rho * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] <= M1 * (1-b[0]), name="P2_BigM1")
    #             m.addConstr(eta_gtw[g,t,w] <= M1 * b[0], name="D_BigM1")
    #             m.addConstr(q_gbar[g] <= M1 * (1-b[1]), name="P_BigM2")
    #             m.addConstr(phi_g[g] <= M1 * b[1], name="D_BigM2")
    #             m.addConstr(q_gtw[g,t,w] <= M1 * (1-b[2]), name="P_BigM3")
    #             m.addConstr(psi_gtw[g,t,w] <= M1 * b[2], name="D_BigM3")
    #             m.addConstr(epsilon_gtw[g,t,w] <= M1 * (1-b[3]), name="P_BigM4")
    #             m.addConstr(l_gtw[g,t,w] <= M1 * b[3], name="D_BigM4")
    #             m.addConstr(psi_gw[g,w] <= M1 * (1-b[4]), name="P_BigM5")
    #             m.addConstr(Theta_gw[g,w] <= M1 * b[4], name="D_BigM5")
    #             m.addConstr(psi_gw[g,w] + Z_gtw + sigma_g[g] >= 0, name="P1_BigM6")
    #             m.addConstr(psi_gw[g,w] + Z_gtw + sigma_g[g] <= M2 * (1-b[5]), name="P2_BigM6")
    #             m.addConstr(delta_gtw[g,t,w] <= M2 * b[5], name="D_BigM6")
    

#     #Big M constraints that only target the conventional generators
#     for gc in range(GC):
#         for t in range(T): 
#             for w in range(W):
#                 m.addConstr(R_gtw[gc,t,w] <= M1 * (1-b[6]), name="P_BigM7")
#                 m.addConstr(muR_gtw[gc,t,w] <= M1 * b[6], name="D_BigM7")
#                 m.addConstr(C_gtw[gc,t,w] <= M1 * (1-b[7]), name="P_BigM8")
#                 m.addConstr(muC_gtw[gc,t,w] <= M1 * b[7], name="D_BigM8")

    # #Writing out the objective function
    # for g in range(G):
    #     for gc in range(GC):
    #         for t in range(T): 
    #             for w in range(W):
    #                 objective = pi_w[w] * tau_t[t] * (beta_w[w] * d_tw[t,w] - (1/2) * alpha_t[t] * d_tw[t,w]**2 - 
    #                             (c_g[g] * q_gtw[g,t,w] - P_CO2*(R_gtw[gc,t,w] - C_gtw[gc,t,w])))-i_g[g] * q_gbar[g]            

    # m.setObjective(objective, GRB.MAXIMIZE)

    # #solving the model
    # m.update()    
    # m.optimize()

#     #Part that seems to not be working as intended, idea for why this is happening is that the model is made form multiple sets this time.

#     if m.Status == GRB.INFEASIBLE:
#         m.computeIIS()
#         m.write('iismodel.ilp')

#         # Print out the IIS constraints and variables
#         print('\nThe following constraints and variables are in the IIS:')
#         for c in m.getConstrs():
#             if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')

#         for v in m.getVars():
#             if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
#             if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')

#     if m.Status == GRB.OPTIMAL:
#         print('\nOptimal objective: %g' % m.objVal)
#         print('Quantity:', q_gtw[g,t,w].x)
#         print('New Capacity:', q_gbar[g].x)
#         print('Emission:', epsilon_gtw[g,t,w].x)
#         print('Demand:', d_tw[t,w].x)
#         print('Revenue:', R_gtw[gc,t,w].x)
#         print('Cost:', C_gtw[gc,t,w].x)
#         print('Auxiliary_Variable1:', sigma_g[g].x)
#         print('Auxiliary_Variable2:', psi_gw[g,w].x)
#         print('Auxiliary_Variable3:', beta_w[w].x)
#         print('Dual-1:', p_tw[t,w].x)
#         print('Dual-2:', eta_gtw[g,t,w].x)
#         print('Dual-5:', phi_g[g].x)
#         print('Dual-6:', delta_gtw[g,t,w].x)
#         print('Dual-7:', Theta_gw[g,w].x)
#         print('Dual-8:', muC_gtw[gc,t,w].x)
#         print('Dual-9:', muR_gtw[gc,t,w].x)
#         print('Dual-10:', l_gtw[g,t,w].x)
#         print('Dual-11:', psi_gtw[g,t,w].x)
#         print('Dual-12:', xi_tw[t,w].x)
#         print('Dual-13:', vR_gtw[g,t,w].x)
#         print('Dual-14:', vC_gtw[g,t,w].x)
        
    # return m

# MIQP_Sets(pi_w, tau_t, v_g, beta_w, alpha_t, c_g, i_g, zeta, rho, q_g0, kappa, P_CO2, ecap, conjectural_variation)

###--------------------------------------------------------------------------------------------------------------------------------------------------------###

#array setup

#np.random.seed(201)

#J, T, W = 3, 5, 2

#a_jtw_mean = np.array([0.291, 0.302, 0.271])
#a_jtw_std = 0.0013
#a_jtw_sigma = np.diag(a_jtw_std * a_jtw_mean)
#a_jtw = np.random.multivariate_normal(mean=a_jtw_mean, cov=a_jtw_sigma, size=T*W).reshape(J, T, W)

#print(a_jtw)  # Verify the shape of the resulting array

###--------------------------------------------------------------------------------------------------------------------------------------------------------###

# RES set model

# import numpy as np
# from gurobipy import GRB
# import gurobipy as gp
# import random

# T = 2 #Number of Timesteps
# W = 3 #Number of Scenarios
# G = 2 #Number of Generators
# GC = 1 #Number of Conventional Generators 
# GR = [0,1] #Generator type (0 for conventional, 1 for renewable)
# RCP = [1,1,1,0.6,0.6] #Capacity factors for the generators, thought it could be used as a range
# total_time = 8760  # Total sum required for each scenario
# time_lb = 3000
# time_ub = 6000
# random.seed(201)

# pi_w = {i: 1/W for i in range(W)}

# tau_tw = {}

# # Populate the nested dictionary with values
# for w in range(W):
#     remaining_sum = total_time
#     tau_timeval = []

#     # Distribute the total sum randomly among T timesteps
#     for t in range(T - 1):
#         value = random.randint(time_lb, min(time_ub, remaining_sum - (T - t - 1) * time_lb))
#         tau_timeval.append(value)
#         remaining_sum -= value

#     # The last value ensures the total sum constraint
#     tau_timeval.append(remaining_sum)

#     # Shuffle the values to make it more random
#     random.shuffle(tau_timeval)

#     for t in range(T):
#         key = (t, w)
#         value = tau_timeval[t]  # Extract the value corresponding to the timestep
#         tau_tw[key] = value

# print(tau_tw)

# Beta_values = [50, 100, 150]
# #Beta_values = [50, 61, 72, 83, 94, 106, 117, 128, 139, 150]

# # Initialize an empty nested dictionary
# beta_tw = {}

# # Populate the nested dictionary with values
# for t in range(T):
#     for w in range(W):
#         key = (t, w)
#         value = Beta_values[w]  # Extract the value corresponding to the timestep
#         beta_tw[key] = value

# #print(beta_tw)

# c_g = [10, 0]
# i_g = [50000, 100000]
# alpha_t = {i: 0.5+0.25*i for i in range(T)}
# rho_gtw = {(g, t, w): 1 for g in range(G) for t in range(T) for w in range(W)}
# q_g0 = [10, 10] 
# kappa = 0.4 
# Phi = 0
# conjectural_t = {i: -alpha_t[i]*(1+Phi) for i in range(T)} #Not sure if these parameters are actually parameters

# #beta = 1 #Not sure if these parameters are actually parameters
# #v_g = {i: 1/G for i in range(G)}

# def MIQP_RES(pi_w, tau_tw, beta_tw, alpha_t, c_g, i_g, rho_gtw, q_g0, kappa, conjectural_t, T, W, G, GR, GC):

#     m=gp.Model('RES-model')

#     #Defining the primal variables:
#     q_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Quantity")
#     q_gbar=m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name="New_Capacity")
#     d_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Demand")
#     p_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Price")
#     #phi_g=m.addVars(G, lb=0, vtype=GRB.BINARY, name="Auxiliary_Variable")
#     #psi_gw=m.addVars(G, W, lb=0, vtype=GRB.BINARY, name="Auxiliary_Variable")

#     #Defining the Dual Variables:
#     eta_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-1")

#     #Defining the Binary variables:
#     b1 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable1")
#     b2 = m.addVars(G, vtype=GRB.BINARY, name="Binary_Variable2")
#     b3 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable3")

#     #Defining the Big M:
#     M = 75000

#     #Defining the Primary Constraints
#     for t in range(T):
#         for w in range(W):
#             m.addConstr(gp.quicksum(q_gtw[g,t,w] for g in range(G)) == d_tw[t,w], name="1d")
#             m.addConstr(p_tw[t,w] == beta_tw[t,w] - alpha_t[t] * d_tw[t,w], name="9")

#     for w in range(W):
#         m.addConstr(gp.quicksum(tau_tw[t,w] * gp.quicksum(q_gtw[g,t,w] for g in GR) for t in range(T)) 
#                     >= kappa * gp.quicksum(tau_tw[t,w] * d_tw[t,w] for t in range(T)), name="1e")
   
#     #Defining the complementarity constraints
#     for g in range(G):
#         for t in range(T):
#             for w in range(W):
#                 m.addConstr(-pi_w[w] * tau_tw[t,w] * (p_tw[t,w] + conjectural_t[t] * q_gtw[g,t,w] - c_g[g]) + eta_gtw[g,t,w] >= 0, name="BigM1")
#                 m.addConstr(-pi_w[w] * tau_tw[t,w] * (p_tw[t,w] + conjectural_t[t] * q_gtw[g,t,w] - c_g[g]) + eta_gtw[g,t,w] <= M * (1-b1[g,t,w]), name="BigM2")
#                 m.addConstr(q_gtw[g,t,w] <= M * b1[g,t,w], name="BigM3")

#                 m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] >= 0, name="BigM7")
#                 m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] <= M * (1-b3[g,t,w]), name="BigM8")
#                 m.addConstr(eta_gtw[g,t,w] <= M * b3[g,t,w], name="BigM9")

#         m.addConstr(i_g[g] - gp.quicksum(rho_gtw[g,t,w] * eta_gtw[g,t,w] for t in range(T) for w in range(W)) >= 0, name="BigM4")
#         m.addConstr(i_g[g] - gp.quicksum(rho_gtw[g,t,w] * eta_gtw[g,t,w] for t in range(T) for w in range(W)) <= M * (1-b2[g]), name="BigM5")
#         m.addConstr(q_gbar[g] <= M * b2[g], name="BigM6")
 
#     #Writing out the objective function                
#     objective = gp.quicksum(
#     pi_w[w] * tau_tw[t, w] * (beta_tw[t, w] * d_tw[t, w] - (1/2) * alpha_t[t] * d_tw[t, w] ** 2 - gp.quicksum(c_g[g] * q_gtw[g, t, w] for g in range(G)))
#     for w in range(W) for t in range(T)) - gp.quicksum(i_g[g] * q_gbar[g] for g in range(G))

# # Set objective function
#     m.setObjective(objective, GRB.MAXIMIZE)
 
#     m.update()
#     m.optimize()
 
#     if m.Status == GRB.INFEASIBLE:
#         m.computeIIS()
#         # Print out the IIS constraints and variables
#         print('\nThe following constraints and variables are in the IIS:')
#         for c in m.getConstrs():
#             if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')
 
#         for v in m.getVars():
#             if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
#             if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
 
#     if m.Status == GRB.OPTIMAL:
#         m.printAttr('X')

#     return m

# MIQP_RES(pi_w, tau_tw, beta_tw, alpha_t, c_g, i_g, rho_gtw, q_g0, kappa, conjectural_t, T, W, G, GR, GC)

###--------------------------------------------------------------------------------------------------------------------------------------------------------###

# RES set model averse - Constraints BigM14 and 8f are not working when both are included

#I had the same problem with the Theta_gw dual in the single case where constraint 8f also didn't work, but I think it's necessary to have it in the model,
# because when muted it makes the results of the model totally wrong. BigM14 seems to be the better one to mute as the results at least seem to be more reasonable.

# import numpy as np
# from gurobipy import GRB
# import gurobipy as gp
# import random

# T = 5 #Number of Timesteps
# W = 10 #Number of Scenarios
# G = 5 #Number of Generators
# GC = [1,1,1,0,0] #Generator placement (1 for conventional, 0 for renewable)
# GR = [0,0,0,1,1] #Generator placement (0 for conventional, 1 for renewable)
# RCP = [1,1,1,0.6,0.6] #Capacity factors for the generators, thought it could be used as a range
# total_time = 8760  # Total sum required for each scenario
# time_lb = 900
# time_ub = 2750
# random.seed(201)

# pi_w = {i: 1/W for i in range(W)}

# tau_tw = {}

# # Populate the nested dictionary with values
# for w in range(W):
#     remaining_sum = total_time
#     tau_timeval = []

#     # Distribute the total sum randomly among T timesteps
#     for t in range(T - 1):
#         value = random.randint(time_lb, min(time_ub, remaining_sum - (T - t - 1) * time_lb))
#         tau_timeval.append(value)
#         remaining_sum -= value

#     # The last value ensures the total sum constraint
#     tau_timeval.append(remaining_sum)

#     # Shuffle the values to make it more random
#     random.shuffle(tau_timeval)

#     for t in range(T):
#         key = (t, w)
#         value = tau_timeval[t]  # Extract the value corresponding to the timestep
#         tau_tw[key] = value

# #print(tau_tw)

# #Beta_values = [50, 100, 150]
# Beta_values = [50, 61, 72, 83, 94, 106, 117, 128, 139, 150]

# # Initialize an empty nested dictionary
# beta_tw = {}

# # Populate the nested dictionary with values
# for t in range(T):
#     for w in range(W):
#         key = (t, w)
#         value = Beta_values[w]  # Extract the value corresponding to the timestep
#         beta_tw[key] = value

# #print(beta_tw)

# c_g = [50, 50, 50, 0, 0]
# i_g = [50000, 50000, 50000, 100000, 100000]
# alpha_t = {i: 0.2+0.2*i for i in range(T)}
# rho_gtw = {(g, t, w): 1 for g in range(G) for t in range(T) for w in range(W)}
# q_g0 = [10, 10, 10, 10, 10] 
# kappa = 0.4 
# Phi = 0
# conjectural_t = {i: -alpha_t[i]*(1+Phi) for i in range(T)} #Not sure if these parameters are actually parameters

# #beta = 1 #Not sure if these parameters are actually parameters
# v_g = {i: 0.25*i for i in range(G)}

# def MIQP_RES_Averse(pi_w, tau_tw, beta_tw, alpha_t, c_g, i_g, rho_gtw, q_g0, kappa, conjectural_t, T, W, G, GR, v_g):

#     m=gp.Model('RES-model')

#     #Defining the primal variables:
#     q_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Quantity")
#     q_gbar=m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name="New_Capacity")
#     d_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Demand")
#     p_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Price")
#     phi_g=m.addVars(G, lb=0, vtype=GRB.BINARY, name="Auxiliary_Variable")

#     #Defining the Dual Variables:
#     eta_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-1")
#     Theta_gw=m.addVars(G, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-2")
#     psi_gw=m.addVars(G, W, lb=0, vtype=GRB.BINARY, name="Dual-3")

#     #Defining the Binary variables:
#     b1 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable1")
#     b2 = m.addVars(G, vtype=GRB.BINARY, name="Binary_Variable2")
#     b3 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable3")
#     b4 = m.addVars(G, W, vtype=GRB.BINARY, name="Binary_Variable4")
#     b5 = m.addVars(G, W, vtype=GRB.BINARY, name="Binary_Variable5")

#     #Defining the Big M:
#     M = 25000

#     #Defining the Primary Constraints
#     for t in range(T):
#         for w in range(W):
#             m.addConstr(gp.quicksum(q_gtw[g,t,w] for g in range(G)) == d_tw[t,w], name="1d")
#             m.addConstr(p_tw[t,w] == beta_tw[t,w] - alpha_t[t] * d_tw[t,w], name="9")

#     for w in range(W):
#         m.addConstr(gp.quicksum(tau_tw[t,w] * gp.quicksum(q_gtw[g,t,w] for g in GR) for t in range(T)) 
#                     >= kappa * gp.quicksum(tau_tw[t,w] * d_tw[t,w] for t in range(T)), name="1e")
   
#     #Defining the complementarity constraints & second level
#     for g in range(G):
#         for t in range(T):
#             for w in range(W):
#                 m.addConstr(-Theta_gw[g,w] * tau_tw[t,w] * (p_tw[t,w] + conjectural_t[t] * q_gtw[g,t,w] - c_g[g]) + eta_gtw[g,t,w] >= 0, name="BigM1")
#                 m.addConstr(-Theta_gw[g,w] * tau_tw[t,w] * (p_tw[t,w] + conjectural_t[t] * q_gtw[g,t,w] - c_g[g]) + eta_gtw[g,t,w] <= M * (1-b1[g,t,w]), name="BigM2")
#                 m.addConstr(q_gtw[g,t,w] <= M * b1[g,t,w], name="BigM3")

#                 m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] >= 0, name="BigM7")
#                 m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] <= M * (1-b3[g,t,w]), name="BigM8")
#                 m.addConstr(eta_gtw[g,t,w] <= M * b3[g,t,w], name="BigM9")
        
#         #for variables and parameters with purely g
#         m.addConstr(i_g[g] - gp.quicksum(rho_gtw[g,t,w] * eta_gtw[g,t,w] for t in range(T) for w in range(W)) >= 0, name="BigM4")
#         m.addConstr(i_g[g] - gp.quicksum(rho_gtw[g,t,w] * eta_gtw[g,t,w] for t in range(T) for w in range(W)) <= M * (1-b2[g]), name="BigM5")
#         m.addConstr(q_gbar[g] <= M * b2[g], name="BigM6")

#         m.addConstr(gp.quicksum(Theta_gw[g,w] for w in range(W)) == 1, name="8f") #This is the constraint that is not working if muted it works

#         # #for variables and parameters with purely g and omega
#         for w in range(W):
#             m.addConstr(pi_w[w] / (1-v_g[g]) - Theta_gw[g,w] >= 0, name="BigM10")
#             m.addConstr(pi_w[w] / (1-v_g[g]) - Theta_gw[g,w] <= M * (1-b4[g,w]), name="BigM11")
#             m.addConstr(psi_gw[g,w] <= M * b4[g,w], name="BigM12")

#             m.addConstr(gp.quicksum(tau_tw[t,w] * (p_tw[t,w] * q_gtw[g,t,w] - c_g[g] * q_gtw[g,t,w]) for t in range(T))
#                          - i_g[g] * q_gbar[g] + phi_g[g] + psi_gw[g,w] >= 0, name="BigM13")
#             #m.addConstr(gp.quicksum(tau_tw[t,w] * (p_tw[t,w] * q_gtw[g,t,w] - c_g[g] * q_gtw[g,t,w]) for t in range(T))
#                          #- i_g[g] * q_gbar[g] + phi_g[g] + psi_gw[g,w] <= M * (1-b5[g,w]), name="BigM14") #This is the constraint that is not working if muted it works
#             m.addConstr(Theta_gw[g,w] <= M * b5[g,w], name="BigM15")


#     #Writing out the objective function                
#     objective = gp.quicksum(
#     pi_w[w] * tau_tw[t, w] * (beta_tw[t, w] * d_tw[t, w] - (1/2) * alpha_t[t] * d_tw[t, w] ** 2 - gp.quicksum(c_g[g] * q_gtw[g, t, w] for g in range(G)))
#     for w in range(W) for t in range(T)) - gp.quicksum(i_g[g] * q_gbar[g] for g in range(G))

# # Set objective function
#     m.setObjective(objective, GRB.MAXIMIZE)
 
#     m.update()
#     m.optimize()
 
#     if m.Status == GRB.INFEASIBLE:
#         m.computeIIS()
#         # Print out the IIS constraints and variables
#         print('\nThe following constraints and variables are in the IIS:')
#         for c in m.getConstrs():
#             if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')
 
#         for v in m.getVars():
#             if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
#             if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
 
#     if m.Status == GRB.OPTIMAL:
#         m.printAttr('X')

#     return m

# MIQP_RES_Averse(pi_w, tau_tw, beta_tw, alpha_t, c_g, i_g, rho_gtw, q_g0, kappa, conjectural_t, T, W, G, GR, v_g)

###--------------------------------------------------------------------------------------------------------------------------------------------------------###

# RES neutral emission model

# import numpy as np
# from gurobipy import GRB
# import gurobipy as gp
# import random

# T = 5 #Number of Timesteps
# W = 10 #Number of Scenarios
# G = 5 #Number of Generators
# GC = [1,1,1,0,0] #Generator placement (1 for conventional, 0 for renewable)
# GR = [0,0,0,1,1] #Generator placement (0 for conventional, 1 for renewable)
# RCP = [1,1,1,0.6,0.6] #Capacity factors for the generators, thought it could be used as a range
# total_time = 8760  # Total sum required for each scenario
# time_lb = 900
# time_ub = 2750
# random.seed(201)

# pi_w = {i: 1/W for i in range(W)}

# tau_tw = {}

# # Populate the nested dictionary with values
# for w in range(W):
#     remaining_sum = total_time
#     tau_timeval = []

#     # Distribute the total sum randomly among T timesteps
#     for t in range(T - 1):
#         value = random.randint(time_lb, min(time_ub, remaining_sum - (T - t - 1) * time_lb))
#         tau_timeval.append(value)
#         remaining_sum -= value

#     # The last value ensures the total sum constraint
#     tau_timeval.append(remaining_sum)

#     # Shuffle the values to make it more random
#     random.shuffle(tau_timeval)

#     for t in range(T):
#         key = (t, w)
#         value = tau_timeval[t]  # Extract the value corresponding to the timestep
#         tau_tw[key] = value

# #print(tau_tw)

# #Beta_values = [50, 100, 150]
# Beta_values = [50, 61, 72, 83, 94, 106, 117, 128, 139, 150]

# # Initialize an empty nested dictionary
# beta_tw = {}

# # Populate the nested dictionary with values
# for t in range(T):
#     for w in range(W):
#         key = (t, w)
#         value = Beta_values[w]  # Extract the value corresponding to the timestep
#         beta_tw[key] = value

# #print(beta_tw)

# c_g = {g : 10 if GC[g] else 0 for g in range(G)}
# i_g = {g : 25000 if GC[g] else 35000 for g in range(G)}
# alpha_t = {i: 0.4+0.25*i for i in range(T)}
# rho_gtw = {(g, t, w): 0.75 if GC[g] else 0.45
#            for g in range(G) for t in range(T) for w in range(W)}

# #print(rho_gtw)
# q_g0 = {g : 333 if GC[g] else 1150 for g in range(G)} #Initial capacity of conventional = 1000 MW, renewable = 2300 MW
# kappa_w = {i: 0.5 for i in range(W)}
# Phi = -0.8
# conjectural_t = {i: -alpha_t[i]*(1+Phi) for i in range(T)} #Not sure if these parameters are actually parameters
# e_gCap = [10, 10, 10, 10, 10] #Emission cap for generator [ton]
# P_CO2=55 #Price of CO2 per ton [Euros]
# Zeta_g = {g : 1 if GC[g] else 0 for g in range(G)} #Intensity factor for generator [ton/MWh]

# #print(conjectural_t)

# #beta = 1 #Not sure if these parameters are actually parameters
# #v_g = {i: 1/G for i in range(G)}

# def MIQP_RES_Emis(pi_w, tau_tw, beta_tw, alpha_t, c_g, i_g, rho_gtw, q_g0, kappa_w, conjectural_t, T, W, G, GR, GC, e_gCap, P_CO2, Zeta_g):

#     m=gp.Model('RES-model')

#     #Defining the primal variables:
#     q_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Quantity")
#     q_gbar=m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name="New_Capacity")
#     d_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Demand")
#     p_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Price")
#     eps_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Emission")

#     #Defining the Dual Variables:
#     eta_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-1")
#     l_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-2")

#     #Defining the Binary variables:
#     b1 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable1")
#     b2 = m.addVars(G, vtype=GRB.BINARY, name="Binary_Variable2")
#     b3 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable3")
#     b4 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable4")

#     #Defining the Big M:
#     M = 100000

#     #Defining the Primary Constraints
#     for t in range(T):
#         for w in range(W):
#             m.addConstr(gp.quicksum(q_gtw[g,t,w] for g in range(G)) == d_tw[t,w], name="1d")
#             m.addConstr(p_tw[t,w] == beta_tw[t,w] - alpha_t[t] * d_tw[t,w], name="9")
#             for g in (0,1,2): #Only conventional generators should be considered for emissions
#                 m.addConstr(eps_gtw[g,t,w] - Zeta_g[g] * q_gtw[g,t,w] == 0, name="emission_constraint")

#                 #Big M emission constraint
#                 m.addConstr(eps_gtw[g,t,w] <= M * (1-b4[g,t,w]), name="BigMemis1")
#                 m.addConstr(l_gtw[g,t,w] <= M * b4[g,t,w], name="BigMemis2")

#     for w in range(W):
#         m.addConstr(gp.quicksum(tau_tw[t,w] * gp.quicksum(q_gtw[g,t,w] for g in (3,4)) for t in range(T)) #Only renewable generators should be considered global penetration
#                     >= kappa_w[w] * gp.quicksum(tau_tw[t,w] * d_tw[t,w] for t in range(T)), name="1e")
   
#     #Defining the complementarity constraints
#     for g in range(G):
#         for t in range(T):
#             for w in range(W):
#                 m.addConstr(-pi_w[w] * tau_tw[t,w] * (p_tw[t,w] + conjectural_t[t] * q_gtw[g,t,w] - c_g[g]) + eta_gtw[g,t,w] >= 0, name="BigM1")
#                 m.addConstr(-pi_w[w] * tau_tw[t,w] * (p_tw[t,w] + conjectural_t[t] * q_gtw[g,t,w] - c_g[g]) + eta_gtw[g,t,w] <= M * (1-b1[g,t,w]), name="BigM2")
#                 m.addConstr(q_gtw[g,t,w] <= M * b1[g,t,w], name="BigM3")

#                 m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] >= 0, name="BigM7")
#                 m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] <= M * (1-b3[g,t,w]), name="BigM8")
#                 m.addConstr(eta_gtw[g,t,w] <= M * b3[g,t,w], name="BigM9")

#         #for variables and parameters with purely g
#         m.addConstr(i_g[g] - gp.quicksum(rho_gtw[g,t,w] * eta_gtw[g,t,w] for t in range(T) for w in range(W)) >= 0, name="BigM4")
#         m.addConstr(i_g[g] - gp.quicksum(rho_gtw[g,t,w] * eta_gtw[g,t,w] for t in range(T) for w in range(W)) <= M * (1-b2[g]), name="BigM5")
#         m.addConstr(q_gbar[g] <= M * b2[g], name="BigM6")
 
#     #Writing out the objective function - Only conventional generators that should be considered for emissions               
#     objective = gp.quicksum(
#     pi_w[w] * tau_tw[t, w] * (beta_tw[t, w] * d_tw[t, w] - (1/2) * alpha_t[t] * d_tw[t, w] ** 2 - gp.quicksum(c_g[g] * q_gtw[g, t, w] for g in range(G)) 
#      - gp.quicksum(P_CO2 * (eps_gtw[g,t,w] - e_gCap[g]) for g in (0,1,2))) for w in range(W) for t in range(T)) - gp.quicksum(i_g[g] * q_gbar[g] for g in range(G))
#     # only conventional generators should be considered for emissions

# # Set objective function
#     m.setObjective(objective, GRB.MAXIMIZE)
 
#     m.update()
#     m.optimize()
 
#     if m.Status == GRB.INFEASIBLE:
#         m.computeIIS()
#         # Print out the IIS constraints and variables
#         print('\nThe following constraints and variables are in the IIS:')
#         for c in m.getConstrs():
#             if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')
 
#         for v in m.getVars():
#             if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
#             if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
 
#     if m.Status == GRB.OPTIMAL:
#         m.printAttr('X')

#     return m

# MIQP_RES_Emis(pi_w, tau_tw, beta_tw, alpha_t, c_g, i_g, rho_gtw, q_g0, kappa_w, conjectural_t, T, W, G, GR, GC, e_gCap, P_CO2, Zeta_g)


###--------------------------------------------------------------------------------------------------------------------------------------------------------###

# RES set model averse - Constraints BigM14 and 8f are not working when both are included

#I had the same problem with the Theta_gw dual in the single case where constraint 8f also didn't work, but I think it's necessary to have it in the model,
# because when muted it makes the results of the model totally wrong. BigM14 seems to be the better one to mute as the results at least seem to be more reasonable.

import numpy as np
from gurobipy import GRB
import gurobipy as gp
import random

T = 5 #Number of Timesteps
W = 10 #Number of Scenarios
G = 5 #Number of Generators
GC = [1,1,1,0,0] #Generator placement (1 for conventional, 0 for renewable)
GR = [0,0,0,1,1] #Generator placement (0 for conventional, 1 for renewable)
RCP = [1,1,1,0.6,0.6] #Capacity factors for the generators, thought it could be used as a range
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

#Beta_values = [50, 100, 150]
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

c_g = {g : 10 if GC[g] else 0 for g in range(G)}
i_g = {g : 250 if GC[g] else 350 for g in range(G)}
alpha_t = {i: 0.4+0.25*i for i in range(T)}
rho_gtw = {(g, t, w): 0.75 if GC[g] else 0.45
           for g in range(G) for t in range(T) for w in range(W)}

#print(rho_gtw)
q_g0 = {g : 100 if GC[g] else 300 for g in range(G)} #Initial capacity of conventional = 1000 MW, renewable = 2300 MW
kappa_w = {i: 0.5 for i in range(W)}
Phi = 0
conjectural_t = {i: -alpha_t[i]*(1+Phi) for i in range(T)} #Not sure if these parameters are actually parameters
e_gCap = [10, 10, 10, 10, 10] #Emission cap for generator [ton]
P_CO2=55 #Price of CO2 per ton [Euros]
Zeta_g = {g : 1 if GC[g] else 0 for g in range(G)} #Intensity factor for generator [ton/MWh]

#beta = 1 #Not sure if these parameters are actually parameters
v_g = {i: 0.2*i for i in range(G)}

def MIQP_RES_Ave_Emis(pi_w, tau_tw, beta_tw, alpha_t, c_g, i_g, rho_gtw, q_g0, kappa_w, conjectural_t, T, W, G, GR, v_g, e_gCap, P_CO2, Zeta_g):

    m=gp.Model('RES-model')

    #Defining the primal variables:
    q_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Quantity")
    q_gbar=m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name="New_Capacity")
    d_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Demand")
    p_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Price")
    phi_g=m.addVars(G, lb=0, vtype=GRB.BINARY, name="Auxiliary_Variable")
    eps_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Emission")

    #Defining the Dual Variables:
    eta_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-1")
    Theta_gw=m.addVars(G, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-2")
    psi_gw=m.addVars(G, W, lb=0, vtype=GRB.BINARY, name="Dual-3")
    xi_tw=m.addVars(T, W, lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual-4")

    #Defining the Binary variables:
    b1 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable1")
    b2 = m.addVars(G, vtype=GRB.BINARY, name="Binary_Variable2")
    b3 = m.addVars(G, T, W, vtype=GRB.BINARY, name="Binary_Variable3")
    b4 = m.addVars(G, W, vtype=GRB.BINARY, name="Binary_Variable4")
    b5 = m.addVars(G, W, vtype=GRB.BINARY, name="Binary_Variable5")


    #Defining the Big M:
    M = 50000

    #Defining the Primary Constraints
    for t in range(T):
        for w in range(W):
            m.addConstr(gp.quicksum(q_gtw[g,t,w] for g in range(G)) == d_tw[t,w], name="1d")
            m.addConstr(p_tw[t,w] == beta_tw[t,w] - alpha_t[t] * d_tw[t,w], name="9")

            for g in (0,1,2): #Only conventional generators should be considered for emissions
                m.addConstr(eps_gtw[g,t,w] - Zeta_g[g] * q_gtw[g,t,w] == 0, name="emission_constraint")

                #m.addConstr(-Theta_gw[g,w] * pi_w[w] * tau_tw[t,w] * P_CO2 - xi_tw[t,w] == 0)
                #Thinking that there might be some extra constraint regarding the partial derivative to the emission (one above)) though when used it makes solution infeasible/unbounded.

    for w in range(W):
        m.addConstr(gp.quicksum(tau_tw[t,w] * gp.quicksum(q_gtw[g,t,w] for g in (3,4)) for t in range(T)) 
                    >= kappa_w[w] * gp.quicksum(tau_tw[t,w] * d_tw[t,w] for t in range(T)), name="1e")
   
    #Defining the complementarity constraints & second level
    for g in range(G):
        for t in range(T):
            for w in range(W):
                m.addConstr(-Theta_gw[g,w] * pi_w[w] * tau_tw[t,w] * (p_tw[t,w] + conjectural_t[t] * q_gtw[g,t,w] - c_g[g])
                             + eta_gtw[g,t,w] + Zeta_g[g] * xi_tw[t,w] >= 0, name="BigM1")
                m.addConstr(-Theta_gw[g,w] * pi_w[w] * tau_tw[t,w] * (p_tw[t,w] + conjectural_t[t] * q_gtw[g,t,w] - c_g[g])
                             + eta_gtw[g,t,w] + Zeta_g[g] * xi_tw[t,w] <= M * (1-b1[g,t,w]), name="BigM2")
                m.addConstr(q_gtw[g,t,w] <= M * b1[g,t,w], name="BigM3")

                m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] >= 0, name="BigM7")
                m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] <= M * (1-b3[g,t,w]), name="BigM8")
                m.addConstr(eta_gtw[g,t,w] <= M * b3[g,t,w], name="BigM9")
        
        #for variables and parameters with purely g
        m.addConstr(i_g[g] - gp.quicksum(rho_gtw[g,t,w] * eta_gtw[g,t,w] for t in range(T) for w in range(W)) >= 0, name="BigM4")
        m.addConstr(i_g[g] - gp.quicksum(rho_gtw[g,t,w] * eta_gtw[g,t,w] for t in range(T) for w in range(W)) <= M * (1-b2[g]), name="BigM5")
        m.addConstr(q_gbar[g] <= M * b2[g], name="BigM6")

        m.addConstr(gp.quicksum(Theta_gw[g,w] for w in range(W)) == 1, name="8f") #This is the constraint that is not working if muted it works

        # #for variables and parameters with purely g and omega
        for w in range(W):
            #m.addConstr(pi_w[w] / (1-v_g[g]) - Theta_gw[g,w] == 0, name="BigM10") Constraints BigM10 - BigM12 should probably be exchanged with this one.

            m.addConstr(pi_w[w] / (1-v_g[g]) - Theta_gw[g,w] >= 0, name="BigM10")
            m.addConstr(pi_w[w] / (1-v_g[g]) - Theta_gw[g,w] <= M * (1-b4[g,w]), name="BigM11")
            m.addConstr(psi_gw[g,w] <= M * b4[g,w], name="BigM12")

            m.addConstr(gp.quicksum(pi_w[w] * tau_tw[t, w] * (p_tw[t,w] * q_gtw[g,t,w] - gp.quicksum(c_g[g] * q_gtw[g, t, w] for g in range(G)) 
                      - gp.quicksum(P_CO2 * (eps_gtw[g,t,w] - e_gCap[g]) for g in (0,1,2))) for w in range(W) for t in range(T))
                      - gp.quicksum(i_g[g] * q_gbar[g] for g in range(G)) + phi_g[g] + psi_gw[g,w] >= 0, name="BigM13")
            # m.addConstr(gp.quicksum(pi_w[w] * tau_tw[t, w] * (p_tw[t,w] * q_gtw[g,t,w] - gp.quicksum(c_g[g] * q_gtw[g, t, w] for g in range(G)) 
            #           - gp.quicksum(P_CO2 * (eps_gtw[g,t,w] - e_gCap[g]) for g in (0,1,2))) for w in range(W) for t in range(T))
            #           - gp.quicksum(i_g[g] * q_gbar[g] for g in range(G)) + phi_g[g] + psi_gw[g,w] <= M * (1-b5[g,w]), name="BigM14") #This is the constraint that is not working if muted it works
            m.addConstr(Theta_gw[g,w] <= M * b5[g,w], name="BigM15")


    #Writing out the objective function                
    objective = gp.quicksum(
    pi_w[w] * tau_tw[t, w] * (beta_tw[t, w] * d_tw[t, w] - (1/2) * alpha_t[t] * d_tw[t, w] ** 2 - gp.quicksum(c_g[g] * q_gtw[g, t, w] for g in range(G)) 
     - gp.quicksum(P_CO2 * (eps_gtw[g,t,w] - e_gCap[g]) for g in (0,1,2))) for w in range(W) for t in range(T)) - gp.quicksum(i_g[g] * q_gbar[g] for g in range(G))
    # only conventional generators should be considered for emissions

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

    return m

MIQP_RES_Ave_Emis(pi_w, tau_tw, beta_tw, alpha_t, c_g, i_g, rho_gtw, q_g0, kappa_w, conjectural_t, T, W, G, GR, v_g, e_gCap, P_CO2, Zeta_g)

###--------------------------------------------------------------------------------------------------------------------------------------------------------###