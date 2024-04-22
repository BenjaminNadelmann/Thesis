# Our model

import numpy as np
from gurobipy import GRB
import gurobipy as gp
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

### Initialize parameter values

G = 3 #Number of Conventional Generators
J = 3 #Number of Renewable Generators
T = 10 #Number of Timesteps
W = 10 #Number of Scenarios
np.random.seed(221)

pi_w = {i: 1/W for i in range(W)}

tau_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# tau_values = [24, 48, 96] #Weekly timetable
#tau_values = [120, 240, 360] #Monthly timetable
#tau_values = [2920, 2920, 2920]
#Beta_values = [50, 61, 72, 83, 94, 106, 117, 128, 139, 150]

# Initialize an empty nested dictionary
tau_tw = {}

# Populate the nested dictionary with values
for w in range(W):
    for t in range(T):
        key = (t, w)
        value = tau_values[t]  # Extract the value corresponding to the timestep
        tau_tw[key] = value

#print(tau_tw)

Beta_tw_mean = np.array([150])
Beta_tw_std = 2.5
Beta_tw_sigma = np.diag(Beta_tw_mean * Beta_tw_std)
Beta_tw = multivariate_normal(mean = Beta_tw_mean, cov = Beta_tw_sigma).rvs(size =T * W).reshape(T, W)
# print(Beta_tw)

flat_beta_tw = Beta_tw.flatten()

beta_tw = {(i // Beta_tw.shape[1], i % Beta_tw.shape[1]): flat_beta_tw[i] for i in range(len(flat_beta_tw))}
# print(beta_tw)


c_g = {0: 23, 1: 28, 2: 17}

i_g = {0: 20, 1: 13, 2: 25}

i_j = {0: 45, 1: 35, 2: 40}

alpha_t = {t: 0.005 for t in range(T)}

Rho_gtw_mean =  np.array([0.75, 0.60, 0.85])
Rho_gtw_std = 0.001
Rho_gtw_sigma = np.diag(Rho_gtw_mean * Rho_gtw_std)
Rho_gtw = multivariate_normal(mean = Rho_gtw_mean, cov = Rho_gtw_sigma).rvs(size =T * W).reshape(G, T, W)
# print(Rho_gtw)

flat_rho_gtw = Rho_gtw.flatten()

rho_gtw = {(i // Rho_gtw.shape[1] // Rho_gtw.shape[2], (i // Rho_gtw.shape[2]) % Rho_gtw.shape[1], i % Rho_gtw.shape[2]): flat_rho_gtw[i] for i in range(len(flat_rho_gtw))}
# print(rho_gtw)

Rho_jtw_mean =  np.array([0.40, 0.25, 0.33])
Rho_jtw_std = 0.001
Rho_jtw_sigma = np.diag(Rho_jtw_mean * Rho_jtw_std)
Rho_jtw = multivariate_normal(mean = Rho_jtw_mean, cov = Rho_jtw_sigma).rvs(size =T * W).reshape(G, T, W)
# print(Rho_jtw)

flat_rho_jtw = Rho_jtw.flatten()

rho_jtw = {(i // Rho_jtw.shape[1] // Rho_jtw.shape[2], (i // Rho_jtw.shape[2]) % Rho_jtw.shape[1], i % Rho_jtw.shape[2]): flat_rho_jtw[i] for i in range(len(flat_rho_jtw))}
# print(rho_jtw)

# q_g0 = {g : 500 for g in range(G)} 
q_g0 = {0: 2030, 1: 2460, 2: 2275}

# q_j0 = {j : 1000 for j in range(J)}
q_j0 = {0: 4750, 1: 5370, 2: 5020} 

kappa_rw = {w: 0.45 for w in range(W)} #Check for percentage of renewable penetration
kappa_cw = {w: 0.45 for w in range(W)} #Check for percentage of conventional penetration
Phi = -0
conjectural_t = {i: -alpha_t[i]*(1+Phi) for i in range(T)} #Not sure if these parameters are actually parameters

P_CO2_max = 30
#P_CO2_tw = {(t, w): 180 for t in range(T) for w in range(W)}

P_CO2_values = [75, 75, 75, 75, 75, 75, 75, 75, 75, 75]

P_CO2_tw = {}

# Populate the nested dictionary with values
for t in range(T):
    for w in range(W):
        key = (t, w)
        value = P_CO2_values[w]  # Extract the value corresponding to the timestep
        P_CO2_tw[key] = value

epsilon_max_g = {0: 10000, 1: 12000, 2: 11000}
epsilon_cap_g = {0: 10000, 1: 12000, 2: 11000}

Xi_gtw_mean =  np.array([0.7, 0.9, 0.8])
Xi_gtw_std = 0.001
Xi_gtw_sigma = np.diag(Xi_gtw_mean * Xi_gtw_std)
Xi_gtw = multivariate_normal(mean = Xi_gtw_mean, cov = Xi_gtw_sigma).rvs(size =T * W).reshape(G, T, W)
# print(Xi_gtw)

flat_xi_gtw = Xi_gtw.flatten()

xi_gtw = {(i // Xi_gtw.shape[1] // Xi_gtw.shape[2], (i // Xi_gtw.shape[2]) % Xi_gtw.shape[1], i % Xi_gtw.shape[2]): flat_xi_gtw[i] for i in range(len(flat_xi_gtw))}
# print(xi_gtw)

v = 0.2
beta = 0.5 #When == 0 then the model is risk neutral and the slackness conditions 7 and 15 aren't violated


###------------------------------------------------------------------Making Optimization model-------------------------------------------------------------------###

def Model(pi_w, tau_tw, beta_tw, c_g, i_g, i_j, alpha_t, rho_gtw, rho_jtw, q_g0, q_j0, kappa_rw, kappa_cw, conjectural_t,
           P_CO2_max, P_CO2_tw, epsilon_max_g, epsilon_cap_g, xi_gtw, v, beta, G, J, T, W):
    m = gp.Model("SetsModel")

    ### Primal variables

    ## For Conventional Generators (g)
    q_gtw = m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name = "q_gtw")
    p_tw = m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name = "p_tw")
    d_tw = m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name = "d_tw")
    E_gtw = m.addVars(G, T, W, vtype=GRB.CONTINUOUS, name = "E_gtw")
    q_gbar = m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name = "q_gbar")
    E_g = m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name = "E_g")
    gamma_gw = m.addVars(G, W, lb=0, vtype=GRB.CONTINUOUS, name = "gamma_gw")
    sigma_g = m.addVars(G, vtype=GRB.CONTINUOUS, name = "sigma_g")

    ## For Renewable Generators (j)
    q_jtw = m.addVars(J, T, W, lb=0, vtype=GRB.CONTINUOUS, name = "q_jtw")
    q_jbar = m.addVars(J, lb=0, vtype=GRB.CONTINUOUS, name = "q_jbar")
    gamma_jw = m.addVars(J, W, lb=0, vtype=GRB.CONTINUOUS, name = "gamma_jw")
    sigma_j = m.addVars(J, vtype=GRB.CONTINUOUS, name = "sigma_j")

    ### Dual variables

    ## For Conventional Generators (g)
    eta_gtw = m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name = "eta_gtw")
    nu_min_gtw = m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name = "nu_min_gtw")
    nu_max_gtw = m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name = "nu_max_gtw")
    zeta_gtw = m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name = "zeta_gtw")
    mu_gtw = m.addVars(G, T, W, vtype=GRB.CONTINUOUS, name = "mu_gtw")
    Delta_gw = m.addVars(G, W, lb=0, vtype=GRB.CONTINUOUS, name = "Delta_gw")
    nu_g_min = m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name = "nu_g_min")
    nu_g_cap = m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name = "nu_g_cap")
    theta_gw = m.addVars(G, W, lb=0, vtype=GRB.CONTINUOUS, name = "theta_gw")
    phi_g = m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name = "phi_g")

    ## For Renewable Generators (j)
    eta_jtw = m.addVars(J, T, W, lb=0, vtype=GRB.CONTINUOUS, name = "eta_jtw")
    Delta_jw = m.addVars(J, W, lb=0, vtype=GRB.CONTINUOUS, name = "Delta_jw")
    theta_jw = m.addVars(J, W, lb=0, vtype=GRB.CONTINUOUS, name = "theta_jw")
    zeta_jtw = m.addVars(J, T, W, lb=0, vtype=GRB.CONTINUOUS, name = "zeta_jtw")
    phi_j = m.addVars(J, lb=0, vtype=GRB.CONTINUOUS, name = "phi_j")

    ### Binary variables

    ## For Conventional Generators (g)
    psi_bar_gtw = m.addVars(G, T, W, vtype=GRB.BINARY, name = "psi_bar_gtw")
    psi_min_gtw = m.addVars(G, T, W, vtype=GRB.BINARY, name = "psi_min_gtw")
    psi_max_gtw = m.addVars(G, T, W, vtype=GRB.BINARY, name = "psi_max_gtw")
    psi_q_gtw = m.addVars(G, T, W, vtype=GRB.BINARY, name = "psi_g_gtw")
    psi_g_bar = m.addVars(G, vtype=GRB.BINARY, name = "psi_g_bar")
    psi_c_gw = m.addVars(G, W, vtype=GRB.BINARY, name = "psi_c_gw")
    psi_fA_gtw = m.addVars(G, T, W, vtype=GRB.BINARY, name = "psi_fA_gtw")
    psi_g_min = m.addVars(G, vtype=GRB.BINARY, name = "psi_g_min")
    psi_g_cap = m.addVars(G, vtype=GRB.BINARY, name = "psi_g_cap")

    ## For Renewable Generators (j)
    psi_bar_jtw = m.addVars(J, T, W, vtype=GRB.BINARY, name = "psi_bar_jtw")
    psi_q_jtw = m.addVars(J, T, W, vtype=GRB.BINARY, name = "psi_q_jtw")
    psi_j_bar = m.addVars(J, vtype=GRB.BINARY, name = "psi_j_bar")
    psi_c_jw = m.addVars(J, W, vtype=GRB.BINARY, name = "psi_c_jw")
    psi_fB_jtw = m.addVars(J, T, W, vtype=GRB.BINARY, name = "psi_fB_jtw")

    ### Big-M values - Values work for each timestep and scenario, not the sum of them 

    ## Primary
    M_p1 = 7500          #Reasoning: Includes the balance between capacity and production for each timestep and scenario 
    M_p2 = 18000         #Reasoning: With the max E_gtw of 6000 and a cap of 5000 then 12000 should be sufficient #only used for conventional generators
    M_p3 = 18000         #Reasoning: With an emission cap of 5000 then 6000 seems fullfilling #only used for conventional generators
    M_p4 = 7500          #Reasoning: Production quantity - can be quite low as it's the maximum for each timestep and scenario    
    M_p5 = 21000         #Reasoning: As the new capacity shouldn't be too high and the initial max is 5370 then 7000 should be sufficient
    M_p6 = 870000        #Reasoning: Not really sure but as it have an impact on the profit then through trial & error 3250000 should be sufficient #Results differ greatly depending on value
    M_p7 = 7000          #Reasoning: Value doesn't really change much no matter what input is given. Probably because of slackness condition violation
    M_p8 = 180000         #Reasoning: Started at 12000 ended at 2500 to not have any numerical mistakes #only used for conventional generators
    M_p9 = 180000         #Reasoning: With an emission cap of 5000 then 6000 seems fullfilling #only used for conventional generators
    
    ## Dual - 3 times the primary values besides, M_d5 which is 100000 (needed for the dual of investment costs), and m_d6 which doesn't need the higher dual value
    M_d1 = 20000        
    M_d2 = 18000        #only used for conventional generators
    M_d3 = 18000        #only used for conventional generators
    M_d4 = 20000      
    M_d5 = 60000        
    M_d6 = 45000     
    M_d7 = 20000    
    M_d8 = 18000        #only used for conventional generators
    M_d9 = 18000        #only used for conventional generators

    # M_p1, M_p2, M_p3, M_p4, M_p5, M_p6, M_p7, M_p8, M_p9 = 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000
    # M_d1, M_d2, M_d3, M_d4, M_d5, M_d6, M_d7, M_d8, M_d9 = 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000

###-----------------------------------------------------------Introducing Constraints-----------------------------------------------------------###


    ### Upper level Constraints
    for t in range(T):
        for w in range(W):
            m.addConstr(gp.quicksum(q_gtw[g,t,w] for g in range(G)) + gp.quicksum(q_jtw[j,t,w] for j in range(J)) == d_tw[t,w] , name="1.13b")
            m.addConstr(p_tw[t,w] == beta_tw[t,w] - alpha_t[t] * d_tw[t,w], name="1.13d")

    for w in range(W):
        m.addConstr(gp.quicksum(tau_tw[t,w] * q_jtw[j,t,w] for j in range(J) for t in range(T)) >= kappa_rw[w] * gp.quicksum(tau_tw[t,w] * d_tw[t,w] for t in range(T)), name="1.13c")

        #Constraint to check the global penetration of conventional generators    
        # m.addConstr(gp.quicksum(tau_tw[t,w] * q_gtw[g,t,w] for g in range(G) for t in range(T)) >= kappa_cw[w] * gp.quicksum(tau_tw[t,w] * d_tw[t,w] for t in range(T)), name="1.13c") 

    ### Lower level Constraints for Conventional Generators (g)
    ## KKT constraints 

    for g in range(G):
        for t in range(T):
            for w in range(W):
                m.addConstr(-(1 - beta) * (pi_w[w] * tau_tw[t,w] * (p_tw[t,w] + (conjectural_t[t] * q_gtw[g,t,w]) - c_g[g]) - pi_w[w] * tau_tw[t,w] * P_CO2_tw[t,w] * xi_gtw[g,t,w]) - \
                            Delta_gw[g,w] * (tau_tw[t,w] * (p_tw[t,w] + (conjectural_t[t] * q_gtw[g,t,w]) - c_g[g]) - tau_tw[t,w] * P_CO2_tw[t,w] * xi_gtw[g,t,w]) + \
                                eta_gtw[g,t,w] - zeta_gtw[g,t,w] - mu_gtw[g,t,w] * xi_gtw[g,t,w] == 0, name="1.5a")

                m.addConstr(mu_gtw[g,t,w] - nu_min_gtw[g,t,w] + nu_max_gtw[g,t,w] == 0, name="1.5c") 
 
                # m.addConstr((beta - 1) * (pi_w[w] * tau_tw[t,w] * P_CO2_tw[t,w] - P_CO2_max) - Delta_gw[g,w] * (tau_tw[t,w] * P_CO2_tw[t,w] - P_CO2_max) + \
                #             (1 / abs(T)) * mu_gtw[g,t,w] - nu_g_min[g] + nu_g_cap[g] == 0, name="1.5d")
                
                m.addConstr(-(1 - beta) * gp.quicksum(pi_w[w] * tau_tw[t,w] * P_CO2_tw[t,w] - P_CO2_max for w in range(W)) - gp.quicksum(Delta_gw[g,w] * tau_tw[t,w] * \
                             P_CO2_tw[t,w] - P_CO2_max for w in range(W)) + (1 / abs(T)) * mu_gtw[g,t,w] - nu_g_min[g] + nu_g_cap[g] == 0, name="1.5d")

                m.addConstr(xi_gtw[g,t,w] * q_gtw[g,t,w] - (1 / abs(T)) * E_g[g] - E_gtw[g,t,w] == 0, name="1.5g") 

        for w in range(W):
            m.addConstr(((beta * pi_w[w]) / (1 - v)) - Delta_gw[g,w] - theta_gw[g,w] == 0, name="1.5e")

            m.addConstr(- (1 - beta) * i_g[g] - gp.quicksum(Delta_gw[g,w] * i_g[g] for w in range(W)) + \
                         gp.quicksum( eta_gtw[g,t,w] * rho_gtw[g,t,w] for t in range(T) for w in range(W)) - phi_g[g] == 0, name="1.5b")

        m.addConstr(- beta + gp.quicksum(Delta_gw[g,w] for w in range(W)) == 0, name="1.5f")

    ## Big-M complementarity constraints

    for g in range(G):
        for t in range(T):
            for w in range(W):
                m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] >= 0, name="1.6a")
                m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] <= M_p1 * (1 - psi_bar_gtw[g,t,w]), name="1.6b") 
                m.addConstr(eta_gtw[g,t,w] <= M_d1 * psi_bar_gtw[g,t,w], name="1.6c")

                m.addConstr(E_gtw[g,t,w] + epsilon_max_g[g] >= 0, name="1.6d")
                m.addConstr(E_gtw[g,t,w] + epsilon_max_g[g] <= M_p2 * (1 - psi_min_gtw[g,t,w]), name="1.6e")  #This Big-M constraint is makes the model infeasible
                m.addConstr(nu_min_gtw[g,t,w] <= M_d2 * psi_min_gtw[g,t,w], name="1.6f")

                m.addConstr(epsilon_max_g[g] - E_gtw[g,t,w] >= 0, name="1.6g")
                m.addConstr(epsilon_max_g[g] - E_gtw[g,t,w] <= M_p3 * (1-psi_max_gtw[g,t,w]), name="1.6h")
                m.addConstr(nu_max_gtw[g,t,w] <= M_d3 * psi_max_gtw[g,t,w], name="1.6i")

                m.addConstr(q_gtw[g,t,w] >= 0, name="1.6j")
                m.addConstr(q_gtw[g,t,w] <= M_p4 * (1 - psi_q_gtw[g,t,w]), name="1.6k")
                m.addConstr(zeta_gtw[g,t,w] <= M_d4 * psi_q_gtw[g,t,w], name="1.6l")

        m.addConstr(q_gbar[g] >= 0, name="1.6m")
        m.addConstr(q_gbar[g] <= M_p5 * (1 - psi_g_bar[g]), name="1.6n")
        m.addConstr(phi_g[g] <= M_d5 * psi_g_bar[g], name="1.6o")
                
        m.addConstr(E_g[g] >= 0, name="1.6v")
        m.addConstr(E_g[g] <= M_p8 * (1 - psi_g_min[g]), name="1.6w")
        m.addConstr(nu_g_min[g] <= M_d8 * psi_g_min[g], name="1.6x")

        m.addConstr(epsilon_cap_g[g] - E_g[g] >= 0, name="1.6y")                            
        m.addConstr(epsilon_cap_g[g] - E_g[g] <= M_p9 * (1 - psi_g_cap[g]), name="1.6z")
        m.addConstr(nu_g_cap[g] <= M_d9 * psi_g_cap[g], name="1.6aa")    

        for w in range(W):

            m.addConstr(gamma_gw[g,w] >= 0, name="1.6p")
            m.addConstr(gamma_gw[g,w] <= M_p6 * (1 - psi_c_gw[g,w]), name="1.6q")
            m.addConstr(theta_gw[g,w] <= M_d6 * psi_c_gw[g,w], name="1.6r")
            
            #The model runs with the following Big-M constraints, but the complementarity slackness conditions gets violated
            m.addConstr((gamma_gw[g,w] + gp.quicksum(tau_tw[t,w] * (p_tw[t,w] * q_gtw[g,t,w] - c_g[g] * q_gtw[g,t,w] + P_CO2_tw[t,w] * \
                        (E_g[g] - xi_gtw[g,t,w] * q_gtw[g,t,w]))  for t in range(T)) - i_g[g] * q_gbar[g] - E_g[g] * P_CO2_max - sigma_g[g]) >= 0, name="1.6s") 
            m.addConstr((gamma_gw[g,w] + gp.quicksum(tau_tw[t,w] * (p_tw[t,w] * q_gtw[g,t,w] - c_g[g] * q_gtw[g,t,w] + P_CO2_tw[t,w] * \
                        (E_g[g] - xi_gtw[g,t,w] * q_gtw[g,t,w]))  for t in range(T)) - i_g[g] * q_gbar[g] - E_g[g] * P_CO2_max - sigma_g[g]) <= M_p7 * (1 - psi_fA_gtw[g,t,w]), name="1.6t") 
            m.addConstr(Delta_gw[g,w] <= M_d7 * psi_fA_gtw[g,t,w], name="1.6u")

    ### Lower level Constraints for Renewable Generators (j)
    ## KKT constraints
    
    for j in range(J):
        for t in range(T):
            for w in range(W):
                m.addConstr(-(1 - beta) * (pi_w[w] * tau_tw[t,w] * (p_tw[t,w] + (conjectural_t[t] * q_jtw[j,t,w]))) - \
                            Delta_jw[j,w] * (tau_tw[t,w] * (p_tw[t,w] + (conjectural_t[t] * q_jtw[j,t,w]))) + eta_jtw[j,t,w] - zeta_jtw[j,t,w] == 0, name="1.10a")

        for w in range(W):
            m.addConstr((beta * pi_w[w]) / (1 - v) - Delta_jw[j,w] - theta_jw[j,w] == 0, name="1.10c")

            m.addConstr(- (1 - beta) * i_j[j] - gp.quicksum(Delta_jw[j,w] * i_j[j] for w in range(W)) + \
                         gp.quicksum( eta_jtw[j,t,w] * rho_jtw[j,t,w] for t in range(T) for w in range(W)) - phi_j[j] == 0, name="1.5b")
# 
        m.addConstr(- beta + gp.quicksum(Delta_jw[j,w] for w in range(W)) == 0, name="1.10d")

    ## Big-M complementarity constraints

    for j in range(J):
        for t in range(T):
            for w in range(W):
                m.addConstr(rho_jtw[j,t,w] * (q_j0[j] + q_jbar[j]) - q_jtw[j,t,w] >= 0, name="1.11a")
                m.addConstr(rho_jtw[j,t,w] * (q_j0[j] + q_jbar[j]) - q_jtw[j,t,w] <= M_p1 * (1-psi_bar_jtw[j,t,w]), name="1.11b")
                m.addConstr(eta_jtw[j,t,w] <= M_d1 * psi_bar_jtw[j,t,w], name="1.11c")

                m.addConstr(q_jtw[j,t,w] >= 0, name="1.11d")
                m.addConstr(q_jtw[j,t,w] <= M_p4 * (1 - psi_q_jtw[j,t,w]), name="1.11e")
                m.addConstr(zeta_jtw[j,t,w] <= M_d4 * psi_q_jtw[j,t,w], name="1.11f")
        
        m.addConstr(q_jbar[j] >= 0, name="1.11g")
        m.addConstr(q_jbar[j] <= M_p5 * (1 - psi_j_bar[j]), name="1.11h")
        m.addConstr(phi_j[j] <= M_d5 * psi_j_bar[j], name="1.11i")

        for w in range(W):

            m.addConstr(gamma_jw[j,w] >= 0, name="1.11j")
            m.addConstr(gamma_jw[j,w] <= M_p6 * (1 - psi_c_jw[j,w]), name="1.11k")
            m.addConstr(theta_jw[j,w] <= M_d6 * psi_c_jw[j,w], name="1.11l")

            #The model runs with the following Big-M constraints, but the complementarity slackness conditions gets violated
            m.addConstr((gamma_jw[j,w] + gp.quicksum(tau_tw[t,w] * (p_tw[t,w] * q_jtw[j,t,w]) for t in range(T)) - i_j[j] * q_jbar[j] - sigma_j[j]) >= 0, name="1.11m")
            m.addConstr((gamma_jw[j,w] + gp.quicksum(tau_tw[t,w] * (p_tw[t,w] * q_jtw[j,t,w]) for t in range(T)) - i_j[j] * q_jbar[j] - sigma_j[j]) <= M_p7 * (1 - psi_fB_jtw[j,t,w]), name="1.11n")
            m.addConstr(Delta_jw[j,w] <= M_d7 * psi_fB_jtw[j,t,w], name="1.11o") 

    ### Objective function    
    objective = gp.quicksum(pi_w[w] * tau_tw[t,w] * (beta_tw[t, w] * d_tw[t, w] - (1/2) * alpha_t[t] * d_tw[t, w] * d_tw[t, w] + \
    gp.quicksum(P_CO2_tw[t,w] * (E_g[g] - xi_gtw[g,t,w] * q_gtw[g,t,w]) - c_g[g] * q_gtw[g,t,w] for g in range(G))) for w in range(W) for t in range(T)) - \
    gp.quicksum(i_g[g] * q_gbar[g] + E_g[g] * P_CO2_max for g in range(G)) - gp.quicksum(i_j[j] * q_jbar[j] for j in range(J))


    # Set objective function
    m.setObjective(objective, GRB.MAXIMIZE)
 
    # Update and optimize model
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

        ## Checking complementarity slackness conditions
        tolerance = 10 ** -8
        for g in range(G):
            for t in range(T):
                for w in range(W):
                    Slack1 = eta_gtw[g,t,w].X * (rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g].X) - q_gtw[g,t,w].X)
                    if abs(Slack1) > tolerance:
                        print("Complementarity slackness condition 1 is violated", "Comparison:", "Dual:", eta_gtw[g,t,w].X, "Expression:", (rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g].X) - q_gtw[g,t,w].X))
                    
                    Slack2 = nu_min_gtw[g,t,w].X * (E_gtw[g,t,w].X + epsilon_max_g[g])
                    if abs(Slack2) > tolerance:
                        print("Complementarity slackness condition 2 is violated", "Comparison:", "Dual:", nu_min_gtw[g,t,w].X, "Expression:", (E_gtw[g,t,w].X + epsilon_max_g[g]))

                    Slack3 = nu_max_gtw[g,t,w].X * (epsilon_max_g[g] - E_gtw[g,t,w].X)
                    if abs(Slack3) > tolerance:
                        print("Complementarity slackness condition 3 is violated", "Comparison:", "Dual:", nu_max_gtw[g,t,w].X, "Expression:", (epsilon_max_g[g] - E_gtw[g,t,w].X))

                    Slack4 = zeta_gtw[g,t,w].X * (q_gtw[g,t,w].X)
                    if abs(Slack4) > tolerance:
                        print("Complementarity slackness condition 4 is violated", "Comparison:", "Dual:", zeta_gtw[g,t,w].X, "Expression:", (q_gtw[g,t,w].X))

            Slack5 = phi_g[g].X * (q_gbar[g].X)
            if abs(Slack5) > tolerance:
                print("Complementarity slackness condition 5 is violated", "Comparison:", "Dual:", phi_g[g].X, "Expression:", (q_gbar[g].X))

            Slack8 = nu_g_min[g].X * (E_g[g].X)
            if abs(Slack8) > tolerance:
                print("Complementarity slackness condition 8 is violated", "Comparison:", "Dual:", nu_g_min[g].X, "Expression:", (E_g[g].X))
                    
            Slack9 = nu_g_cap[g].X * (epsilon_cap_g[g] - E_g[g].X)
            if abs(Slack9) > tolerance:
                print("Complementarity slackness condition 9 is violated", "Comparison:", "Dual:", nu_g_cap[g].X, "Expression:", (epsilon_cap_g[g] - E_g[g].X))

            for w in range(W):
                    
                Slack6 = theta_gw[g,w].X * gamma_gw[g,w].X
                if abs(Slack6) > tolerance:
                    print("Complementarity slackness condition 6 is violated", "Comparison:", "Dual:", theta_gw[g,w].X, "Expression:", gamma_gw[g,w].X)

                #Couldn't get the slackness condition for the profit function to work properly, because of the if statement, but if just printed then the values are 0 a

                Slack7 = Delta_gw[g,w].X * (gamma_gw[g,w].X + gp.quicksum(p_tw[t,w].X * q_gtw[g,t,w].X - c_g[g] * q_gtw[g,t,w].X + P_CO2_tw[t,w] * \
                                        (E_g[g].X - xi_gtw[g,t,w] * q_gtw[g,t,w].X)  for t in range(T)) - i_g[g] * q_gbar[g].X - E_g[g].X * P_CO2_max  - sigma_g[g].X)
                print("Slack7 - CVaR profit function Conventional generators", Slack7, "If value == 0 then satisfied condition")
                # if Slack7 != 0:
                #     print("Complementarity slackness condition 7 is violated")



        for j in range(J):
            for t in range(T):
                for w in range(W):
                    Slack11 = eta_jtw[j,t,w].X * (rho_jtw[j,t,w] * (q_j0[j] + q_jbar[j].X) - q_jtw[j,t,w].X)
                    if abs(Slack11) > tolerance:
                        print("Complementarity slackness condition 11 is violated", "Comparison:", "Dual:", eta_jtw[j,t,w].X, "Expression:", (rho_jtw[j,t,w] * (q_j0[j] + q_jbar[j].X) - q_jtw[j,t,w].X))
                    
                    Slack12 = zeta_jtw[j,t,w].X * (q_jtw[j,t,w].X)
                    if abs(Slack12) > tolerance:
                        print("Complementarity slackness condition 12 is violated", "Comparison:", "Dual:", zeta_jtw[j,t,w].X, "Expression:", q_jtw[j,t,w].X)

            Slack13 = phi_j[j].X * (q_jbar[j].X)
            if abs(Slack13) > tolerance:
                print("Complementarity slackness condition 13 is violated", "Comparison:", "Dual:", phi_j[j].X, "Expression:", q_jbar[j].X)

            for w in range(W):

                Slack14 = theta_jw[j,w].X * gamma_jw[j,w].X
                if abs(Slack14) > tolerance:
                    print("Complementarity slackness condition 14 is violated", "Comparison:", "Dual:", theta_jw[j,w].X, "Expression:", gamma_jw[j,w].X)

                Slack15 = Delta_jw[j,w].X * (gamma_jw[j,w].X + gp.quicksum(p_tw[t,w].X * q_jtw[j,t,w].X for t in range(T)) - i_j[j] * q_jbar[j].X - sigma_j[j].X)
                print("Slack15 - CVaR profit function Renewable generators:", Slack15, "If value == 0 then satisfied condition")
                # if Slack15 != 0:
                #     print("Complementarity slackness condition 15 is violated")

        ## Returning all the relevant variables and duals

        obj = m.getObjective().getValue()

        Quantity_Conventional = np.zeros((G, T, W))
        New_Capacity_Conventional = np.zeros(G)
        PreEmission_Conventional = np.zeros(G)
        Emission_Conventional = np.zeros((G, T, W))
        Prices = np.zeros((T, W))
        Demand = np.zeros((T, W))
        Sigma_Conventional = np.zeros(G)
        Gamma_Conventional = np.zeros((G, W))

        for g in range(G):

            New_Capacity_Conventional[g] = q_gbar[g].X
            PreEmission_Conventional[g] = E_g[g].X
            Sigma_Conventional[g] = sigma_g[g].X    

            for t in range(T):
                for w in range(W):
                    Quantity_Conventional[g, t, w] = q_gtw[g, t, w].X
                    Emission_Conventional[g, t, w] = E_gtw[g, t, w].X
                    Prices[t, w] = p_tw[t, w].X
                    Demand[t, w] = d_tw[t, w].X
                    Gamma_Conventional[g, w] = gamma_gw[g, w].X

        Quantity_Renewable = np.zeros((J, T, W))
        New_Capacity_Renewable = np.zeros(J)
        Sigma_Renewable = np.zeros(J)
        Gamma_Renewable = np.zeros((J, W))

        for j in range(J):

            New_Capacity_Renewable[j] = q_jbar[j].X
            Sigma_Renewable[j] = sigma_j[j].X

            for t in range(T):
                for w in range(W):
                    Quantity_Renewable[j, t, w] = q_jtw[j, t, w].X
                    Gamma_Renewable[j, w] = gamma_jw[j, w].X
    
        ## Objective function checkup    
        Expected_Consumer_Surplus_impact = gp.quicksum(pi_w[w] * tau_tw[t,w] * (beta_tw[t,w] * d_tw[t,w].X - (1/2) * alpha_t[t] * d_tw[t,w].X * d_tw[t,w].X) for t in range(T) for w in range(W))
        Con_CO2_cost = gp.quicksum(pi_w[w] * tau_tw[t,w] * (gp.quicksum(P_CO2_tw[t,w] * (E_g[g].X - xi_gtw[g,t,w] * q_gtw[g,t,w].X) - c_g[g] * q_gtw[g,t,w].X for g in range(G))) for t in range(T) for w in range(W))
        Con_invest = gp.quicksum(i_g[g] * q_gbar[g].X + E_g[g].X * P_CO2_max for g in range(G))
        Ren_invest = gp.quicksum(i_j[j] * q_jbar[j].X for j in range(J))

        print("Expected_Consumer_Surplus:", Expected_Consumer_Surplus_impact * 10 ** -6)
        print("Con Costs & CO2 emission:", Con_CO2_cost * 10 ** -6)
        print("Con investment:", - Con_invest * 10 ** -6)
        print("Ren investment:", - Ren_invest * 10 ** -6)

    return Quantity_Conventional, New_Capacity_Conventional, PreEmission_Conventional, Emission_Conventional, Prices, Demand, \
           Sigma_Conventional, Gamma_Conventional, Quantity_Renewable, New_Capacity_Renewable, Sigma_Renewable, Gamma_Renewable, obj

###---------------------------------------------------------Running model and extracting results------------------------------------------------------------###


Q_C, New_CC, Pre_All, All, Price, Demand, Sig_C, Gam_C, Q_R, New_CR, Sig_R, Gam_R, obj = Model(pi_w, tau_tw, beta_tw, c_g, i_g, i_j, alpha_t, rho_gtw, rho_jtw, \
        q_g0, q_j0, kappa_rw, kappa_cw, conjectural_t, P_CO2_max, P_CO2_tw, epsilon_max_g, epsilon_cap_g, xi_gtw, v, beta, G, J, T, W)


#print("Conventional Quantity produced:", Q_C)
#print("Bought Emission allowance throughout:", All)
print("Production Prices:", Price)
print("Demand:", Demand)
#print("Renewable Quantity produced:", Q_R)
print("Pre-bought Emission Allowance:", Pre_All)
print("New Conventional Capacity:", New_CC)
print("New Renewable Capacity:", New_CR)

sum_QC, sum_QR, sum_All = np.sum(Q_C, axis=(1,2)), np.sum(Q_R, axis=(1,2)), np.sum(All, axis=(1,2))
Average_Price_t, Average_Price_S = np.mean(Price, axis=1), np.mean(Price, axis=0)
Time_QC, Time_QR, Time_All = np.sum(Q_C, axis=(0,2)), np.sum(Q_R, axis=(0,2)), np.sum(All, axis=(0,2))

profit_t = (Average_Price_t * Time_QC) + (Average_Price_t * Time_QR) - (Time_All * 75) - (Time_QC * 20)

print("The objective value is:", obj * 10 ** -6, "mEuro")
print("Quantities produced by each conventional generator", sum_QC)
print("Bought emission allowance (during) for each conventional generator", sum_All)
print("Quantities produced by each renewable generator", sum_QR)

print("Total demand by scenario:", np.sum(Demand, axis=(0)))
print("Average Price by scenario:", Average_Price_S)
print("Average Price over Time:", Average_Price_t)
print("Total conventional quantity produced by scenario:", np.sum(Q_C, axis=(0,1)))
print("Bought emission allowance for each scenario", np.sum(All, axis=(0,1)))
print("Total renewable quantity produced by scenario:", np.sum(Q_R, axis=(0,1)))

print("Profit obtained in each timestep:", profit_t)

###----------------------------------------------------------------------Plotting---------------------------------------------------------------------------###

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))

bar_l0 = ['Con{}'.format(i) for i in range(1, G+1)] + ['Ren{}'.format(j) for j in range(1, J+1)]
bar_l1 = ['Conv'] + ['_Conv'] * (G - 1) + ['Ren'] + ['_Ren'] * (J - 1)
bar_c1 = ['tab:red'] * G + ['tab:blue'] * J
bar_l2 = ['S{}'.format(i) for i in range(1, W+1)]
bar_l3 = ['Con{}'.format(i) for i in range(1, G+1)]

t = np.arange(T)  # Time periods

bottom1 = np.zeros(W)
bottom2 = np.zeros(G)

Quantities = {'Ren': np.sum(Q_R, axis=(0,1)) ,'Conv': np.sum(Q_C, axis=(0,1))}
Emissions = {'Prior': Pre_All, 'During': np.sum(All, axis=(1,2))}

axs[0,0].bar(bar_l0, np.concatenate((sum_QC, sum_QR)), label=bar_l1, color=bar_c1)

axs[1,0].bar(bar_l0, np.concatenate((New_CC, New_CR)), label=bar_l1, color=bar_c1)

for boolean, weight_count in Quantities.items():
    p1 = axs[0,1].bar(bar_l2, weight_count, label=boolean, bottom=bottom1)
    bottom1 += weight_count

for boolean, weight_count in Emissions.items():
    p2 = axs[1,1].bar(bar_l3, weight_count, label=boolean, bottom=bottom2)
    bottom2 += weight_count

axs[0,2].plot(t, Average_Price_t, label='Average Price', marker='o', linestyle='-')

axs[1,2].plot(t, profit_t, label='Average Price', marker='o', linestyle='-')

axs[0,0].set_title("Quantity Produced by Generators")
axs[0,0].set_ylabel('Quantity produced [MWh]')
axs[0,0].legend(title='Generator type', loc="upper left")
axs[0,0].ticklabel_format(style='sci', axis='y', scilimits=(4, 4))

axs[1,0].set_ylabel('New capacity [MW]')
axs[1,0].legend(title='Generator type', loc="upper left")
axs[1,0].ticklabel_format(style='sci', axis='y', scilimits=(3, 3))

axs[0,1].set_title("Quantity Produced by scenario")
axs[0,1].set_ylabel('Quantity produced [MWh]')
axs[0,1].legend(title='Generator type', loc="upper left")
axs[0,1].ticklabel_format(style='sci', axis='y', scilimits=(4, 4))

axs[1,1].set_ylabel('Emission allowances [tons]')
axs[1,1].legend(title='Allowance type', loc="upper left")
axs[1,1].ticklabel_format(style='sci', axis='y', scilimits=(4, 4))

axs[0,2].set_title("Average Price over Time")
axs[0,2].set_ylabel('Average Prices [€]')
axs[0,2].set_xlabel('Time Periods [t]')
axs[0,2].legend(loc="upper left")
axs[0,2].set_xticks(t)

axs[1,2].set_ylabel('Profits [€]')
axs[1,2].set_xlabel('Time Periods [t]')
axs[1,2].legend(loc="upper left")
axs[1,2].set_xticks(t)
axs[1,2].ticklabel_format(style='sci', axis='y', scilimits=(5, 5))

plt.show()
