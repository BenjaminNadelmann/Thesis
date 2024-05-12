# Conventional model only with all constraints and variables

import numpy as np
from gurobipy import GRB
import gurobipy as gp
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time

###----------------------------------------------------Initialize parameter values----------------------------------------------------###

G = 3 #Number of Conventional Generators
T = 3 #Number of Timesteps
W = 3 #Number of Scenarios
np.random.seed(221)

pi_w = {i: 1/W for i in range(W)}

tau_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

tau_tw = {}

for w in range(W):
    for t in range(T):
        key = (t, w)
        value = tau_values[t] 
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


c_g = {0: 20, 1: 31, 2: 15}

i_g = {0: 55, 1: 47, 2: 60} 

alpha_t = {t: 0.005 for t in range(T)}

Rho_gtw_mean =  np.array([0.75, 0.60, 0.85])
Rho_gtw_std = 0.001
Rho_gtw_sigma = np.diag(Rho_gtw_mean * Rho_gtw_std)
Rho_gtw = multivariate_normal(mean = Rho_gtw_mean, cov = Rho_gtw_sigma).rvs(size =T * W).reshape(G, T, W)
# print(Rho_gtw)

flat_rho_gtw = Rho_gtw.flatten()

rho_gtw = {(i // Rho_gtw.shape[1] // Rho_gtw.shape[2], (i // Rho_gtw.shape[2]) % Rho_gtw.shape[1], i % Rho_gtw.shape[2]): flat_rho_gtw[i] for i in range(len(flat_rho_gtw))}
# print(rho_gtw)

q_g0 = {0: 2030, 1: 2460, 2: 2275}

Phi = -0 #Change according to market structure 
conjectural_t = {i: -alpha_t[i]*(1+Phi) for i in range(T)} 

P_CO2_max = 30 

P_CO2_values = [75, 75, 75, 75, 75, 75, 75, 75, 75, 75]

P_CO2_tw = {}

for t in range(T):
    for w in range(W):
        key = (t, w)
        value = P_CO2_values[w] 
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

v = 0.95  
 
# beta = 1 #Comment out if running the model for multiple beta values

###------------------------------------------------------------------Making Optimization model-------------------------------------------------------------------###

def Model(pi_w, tau_tw, beta_tw, c_g, i_g, alpha_t, rho_gtw, q_g0, conjectural_t,
           P_CO2_max, P_CO2_tw, epsilon_max_g, epsilon_cap_g, xi_gtw, v, beta, G, T, W):
    
    print(f"Running Model with beta = {beta}")  #Helps to keep track of how far along the model is
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


    ### Big-M values - Values work for each timestep and scenario, not the sum of them 

    ## Primary
    M_p1 = 200000       
    M_p2 = 50000        #only used for conventional generators
    M_p3 = 50000        #only used for conventional generators
    M_p4 = 100000       
    M_p5 = 100000        
    M_p6 = 500000     
    M_p7 = 400000         
    M_p8 = 50000        #only used for conventional generators
    M_p9 = 50000        #only used for conventional generators
    
    ## Dual
    M_d1 = 100000        
    M_d2 = 50000        #only used for conventional generators
    M_d3 = 50000        #only used for conventional generators
    M_d4 = 100000      
    M_d5 = 100000        
    M_d6 = 100000     
    M_d7 = 100000    
    M_d8 = 50000        #only used for conventional generators
    M_d9 = 50000        #only used for conventional generators

###-----------------------------------------------------------Introducing Constraints-----------------------------------------------------------###


    ### Upper level Constraints without renewable components  
    for t in range(T):
        for w in range(W):
            m.addConstr(gp.quicksum(q_gtw[g,t,w] for g in range(G)) == d_tw[t,w] , name="4.16b")
            m.addConstr(p_tw[t,w] == beta_tw[t,w] - alpha_t[t] * d_tw[t,w], name="4.16d")

    ### Lower level Constraints
    ## KKT constraints 

    for g in range(G):
        for t in range(T):
            for w in range(W):
                m.addConstr(-(1 - beta) * (pi_w[w] * tau_tw[t,w] * (p_tw[t,w] + (conjectural_t[t] * q_gtw[g,t,w]) - c_g[g]) - pi_w[w] * tau_tw[t,w] * P_CO2_tw[t,w] * xi_gtw[g,t,w]) - \
                            Delta_gw[g,w] * (tau_tw[t,w] * (p_tw[t,w] + (conjectural_t[t] * q_gtw[g,t,w]) - c_g[g]) - tau_tw[t,w] * P_CO2_tw[t,w] * xi_gtw[g,t,w]) + \
                                eta_gtw[g,t,w] - zeta_gtw[g,t,w] - mu_gtw[g,t,w] * xi_gtw[g,t,w] == 0, name="4.6a")

                m.addConstr(mu_gtw[g,t,w] - nu_min_gtw[g,t,w] + nu_max_gtw[g,t,w] == 0, name="4.6c") 
                
                m.addConstr(-(1 - beta) * gp.quicksum(pi_w[w] * tau_tw[t,w] * P_CO2_tw[t,w] - P_CO2_max for w in range(W)) - gp.quicksum(Delta_gw[g,w] * (tau_tw[t,w] * \
                             P_CO2_tw[t,w] - P_CO2_max) for w in range(W)) + (1 / abs(T)) * mu_gtw[g,t,w] - nu_g_min[g] + nu_g_cap[g] == 0, name="4.6d")

                m.addConstr(xi_gtw[g,t,w] * q_gtw[g,t,w] - (1 / abs(T)) * E_g[g] - E_gtw[g,t,w] == 0, name="4.6g") 

        for w in range(W):
            m.addConstr(((beta * pi_w[w]) / (1 - v)) - Delta_gw[g,w] - theta_gw[g,w] == 0, name="4.6e")

            m.addConstr( (1 - beta) * i_g[g] + gp.quicksum(Delta_gw[g,w] * i_g[g] for w in range(W)) - \
                 gp.quicksum( eta_gtw[g,t,w] * rho_gtw[g,t,w] for t in range(T) for w in range(W)) - phi_g[g] == 0, name="4.6b")

        m.addConstr(- beta + gp.quicksum(Delta_gw[g,w] for w in range(W)) == 0, name="4.6f")

    ## Big-M complementarity constraints

    for g in range(G):
        for t in range(T):
            for w in range(W):
                m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] >= 0, name="4.7a")
                m.addConstr(rho_gtw[g,t,w] * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] <= M_p1 * (1 - psi_bar_gtw[g,t,w]), name="4.7b")  
                m.addConstr(eta_gtw[g,t,w] <= M_d1 * psi_bar_gtw[g,t,w], name="4.7c")

                m.addConstr(E_gtw[g,t,w] + epsilon_max_g[g] >= 0, name="4.7d")
                m.addConstr(E_gtw[g,t,w] + epsilon_max_g[g] <= M_p2 * (1 - psi_min_gtw[g,t,w]), name="4.7e") 
                m.addConstr(nu_min_gtw[g,t,w] <= M_d2 * psi_min_gtw[g,t,w], name="4.7f")

                m.addConstr(epsilon_max_g[g] - E_gtw[g,t,w] >= 0, name="4.7g")
                m.addConstr(epsilon_max_g[g] - E_gtw[g,t,w] <= M_p3 * (1-psi_max_gtw[g,t,w]), name="4.7h")
                m.addConstr(nu_max_gtw[g,t,w] <= M_d3 * psi_max_gtw[g,t,w], name="4.7i")

                m.addConstr(q_gtw[g,t,w] >= 0, name="4.7j")
                m.addConstr(q_gtw[g,t,w] <= M_p4 * (1 - psi_q_gtw[g,t,w]), name="4.7k")
                m.addConstr(zeta_gtw[g,t,w] <= M_d4 * psi_q_gtw[g,t,w], name="4.7l")

        m.addConstr(q_gbar[g] >= 0, name="4.7m")
        m.addConstr(q_gbar[g] <= M_p5 * (1 - psi_g_bar[g]), name="4.7n")
        m.addConstr(phi_g[g] <= M_d5 * psi_g_bar[g], name="4.7o")
                
        m.addConstr(E_g[g] >= 0, name="4.7v")
        m.addConstr(E_g[g] <= M_p8 * (1 - psi_g_min[g]), name="4.7w")
        m.addConstr(nu_g_min[g] <= M_d8 * psi_g_min[g], name="4.7x")

        m.addConstr(epsilon_cap_g[g] - E_g[g] >= 0, name="4.7y")                            
        m.addConstr(epsilon_cap_g[g] - E_g[g] <= M_p9 * (1 - psi_g_cap[g]), name="4.7z")
        m.addConstr(nu_g_cap[g] <= M_d9 * psi_g_cap[g], name="4.7aa")     

        for w in range(W):

            m.addConstr(gamma_gw[g,w] >= 0, name="4.7p")
            m.addConstr(gamma_gw[g,w] <= M_p6 * (1 - psi_c_gw[g,w]), name="4.7q")
            m.addConstr(theta_gw[g,w] <= M_d6 * psi_c_gw[g,w], name="4.7r")
            
            m.addConstr((gamma_gw[g,w] + gp.quicksum(tau_tw[t,w] * (p_tw[t,w] * q_gtw[g,t,w] - c_g[g] * q_gtw[g,t,w] + P_CO2_tw[t,w] * \
                        (E_g[g] - xi_gtw[g,t,w] * q_gtw[g,t,w]))  for t in range(T)) - i_g[g] * q_gbar[g] - E_g[g] * P_CO2_max - sigma_g[g]) >= 0, name="4.7s") 
            m.addConstr((gamma_gw[g,w] + gp.quicksum(tau_tw[t,w] * (p_tw[t,w] * q_gtw[g,t,w] - c_g[g] * q_gtw[g,t,w] + P_CO2_tw[t,w] * \
                        (E_g[g] - xi_gtw[g,t,w] * q_gtw[g,t,w]))  for t in range(T)) - i_g[g] * q_gbar[g] - E_g[g] * P_CO2_max - sigma_g[g]) <= M_p7 * (1 - psi_fA_gtw[g,t,w]), name="4.7t") 
            m.addConstr(Delta_gw[g,w] <= M_d7 * psi_fA_gtw[g,t,w], name="4.7u")


    ### Objective function - 4.16a without renewable components    
    objective = gp.quicksum(pi_w[w] * tau_tw[t,w] * (beta_tw[t, w] * d_tw[t, w] - (1/2) * alpha_t[t] * d_tw[t, w] * d_tw[t, w] + \
    gp.quicksum(P_CO2_tw[t,w] * (E_g[g] - xi_gtw[g,t,w] * q_gtw[g,t,w]) - c_g[g] * q_gtw[g,t,w] for g in range(G))) for w in range(W) for t in range(T)) - \
    gp.quicksum(i_g[g] * q_gbar[g] + E_g[g] * P_CO2_max for g in range(G))


    # Set objective function
    m.setObjective(objective, GRB.MAXIMIZE)
 
    # Update and optimize model
    m.update()
    m.optimize()

###-----------------------------------------------------------Checking the results-----------------------------------------------------------###    
 
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

                Slack7 = Delta_gw[g,w].X * (gamma_gw[g,w].X + gp.quicksum(p_tw[t,w].X * q_gtw[g,t,w].X - c_g[g] * q_gtw[g,t,w].X + P_CO2_tw[t,w] * \
                                        (E_g[g].X - xi_gtw[g,t,w] * q_gtw[g,t,w].X)  for t in range(T)) - i_g[g] * q_gbar[g].X - E_g[g].X * P_CO2_max  - sigma_g[g].X)
                print("Slack7 - CVaR profit function Conventional generators", Slack7, "If value == 0 then satisfied condition")
                # if Slack7 != 0:
                #     print("Complementarity slackness condition 7 is violated")

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

        ## Objective function checkup    
        Expected_Consumer_Surplus_impact = gp.quicksum(pi_w[w] * tau_tw[t,w] * (beta_tw[t,w] * d_tw[t,w].X - (1/2) * alpha_t[t] * d_tw[t,w].X * d_tw[t,w].X) for t in range(T) for w in range(W))
        Con_CO2_cost = gp.quicksum(pi_w[w] * tau_tw[t,w] * (gp.quicksum(P_CO2_tw[t,w] * (E_g[g].X - xi_gtw[g,t,w] * q_gtw[g,t,w].X) - c_g[g] * q_gtw[g,t,w].X for g in range(G))) for t in range(T) for w in range(W))
        Con_invest = gp.quicksum(i_g[g] * q_gbar[g].X + E_g[g].X * P_CO2_max for g in range(G))

        print("Expected_Consumer_Surplus:", Expected_Consumer_Surplus_impact * 10 ** -6)
        print("Con Costs & CO2 emission:", Con_CO2_cost * 10 ** -6)
        print("Con investment:", - Con_invest * 10 ** -6)

    return Quantity_Conventional, New_Capacity_Conventional, PreEmission_Conventional, Emission_Conventional, Prices, Demand, \
           Sigma_Conventional, Gamma_Conventional, obj, Expected_Consumer_Surplus_impact, Con_CO2_cost, Con_invest

###---------------------------------------------------------Running model and extracting results------------------------------------------------------------###

###--------------Model run as a function of risk aversion-----------------###

beta_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
prices = []
demand = []
objectives = []
Quantity = []
New_Capacity = []
DAllowance = []
PAllowance = []
ECSi = []
CC2c = []
Ci = []

Start_timer = time.time()

for beta in beta_values:
    Q_C, New_CC, Pre_All, All, Price, Demand, Sig_C, Gam_C, obj, ECSI, CC2C, CI = Model(pi_w, tau_tw, beta_tw, c_g, i_g, alpha_t, rho_gtw, \
        q_g0, conjectural_t, P_CO2_max, P_CO2_tw, epsilon_max_g, epsilon_cap_g, xi_gtw, v, beta, G, T, W)
    
    prices.append(np.mean(Price, axis=(0,1)))
    demand.append(np.sum(Demand, axis=(0,1)))
    objectives.append(obj)
    Quantity.append(np.sum(Q_C, axis=(1,2)))
    New_Capacity.append(New_CC)
    DAllowance.append(np.sum(All, axis=(1,2)))
    PAllowance.append(Pre_All)
    ECSi.append(ECSI)
    CC2c.append(CC2C)
    Ci.append(CI)

print(prices)
print(demand)
print(objectives)
print(Quantity) 
print(New_Capacity)
print(DAllowance)
print(PAllowance)
print(ECSi)
print(CC2c)
print(Ci)

End_timer = time.time()
elapsed_time = End_timer - Start_timer

#Exporting the results to a text file	

output_file = "output_conv_Courn.txt"
# output_file = "output_conv_Perf.txt"

# Open the file in write mode
with open(output_file, "w") as file:
    file.write(f"Production Prices: {prices}\n")
    file.write(f"Demand: {demand}\n")
    file.write(f"The objective value is: {objectives} Euro\n")
    file.write(f"Quantities produced by each conventional generator: {Quantity}\n")
    file.write(f"New Conventional Capacity: {New_Capacity}\n")
    file.write(f"Bought emission allowance throughout: {DAllowance}\n")
    file.write(f"Pre-bought Emission Allowance: {PAllowance}\n")
    file.write(f"Expected Consumer Surplus impact: {ECSi}\n")
    file.write(f"Conventional Costs & CO2 emission: {CC2c}\n")
    file.write(f"Conventional investment: {Ci}\n")
    file.write(f"Elapsed time for running whole program: {elapsed_time} seconds\n")

print(f"Data written to '{output_file}' successfully.")

###--------------Singular run of the model-----------------###

# Q_C, New_CC, Pre_All, All, Price, Demand, Sig_C, Gam_C, obj = Model(pi_w, tau_tw, beta_tw, c_g, i_g, alpha_t, rho_gtw, \
#         q_g0, conjectural_t, P_CO2_max, P_CO2_tw, epsilon_max_g, epsilon_cap_g, xi_gtw, v, beta, G, T, W)

# print("Conventional Quantity produced:", Q_C)
# print("Bought Emission allowance throughout:", All)
# print("Production Prices:", Price)
# print("Demand:", Demand)
# print("Pre-bought Emission Allowance:", Pre_All)
# print("New Conventional Capacity:", New_CC)

# sum_QC, sum_All = np.sum(Q_C, axis=(1,2)), np.sum(All, axis=(1,2))
# Average_Price_t, Average_Price_S = np.mean(Price, axis=1), np.mean(Price, axis=0)
# Time_QC, Time_All = np.sum(Q_C, axis=(0,2)), np.sum(All, axis=(0,2))

# cg_ar = np.array([c_g[key] for key in sorted(c_g.keys())])
# ig_ar = np.array([i_g[key] for key in sorted(i_g.keys())])

# profit_t = (Average_Price_t * Time_QC) - (Time_All * 75) - (Time_QC * cg_ar) - (New_CC * ig_ar)/T 

# print("The objective value is:", obj * 10 ** -6, "mEuro")
# print("Quantities produced by each conventional generator", sum_QC)
# print("Bought emission allowance (during) for each conventional generator", sum_All)

# print("Total demand by scenario:", np.sum(Demand, axis=(0)))
# print("Average Price by scenario:", Average_Price_S)
# print("Average Price over Time:", Average_Price_t)
# print("Total conventional quantity produced by scenario:", np.sum(Q_C, axis=(0,1)))
# print("Bought emission allowance for each scenario", np.sum(All, axis=(0,1)))

# print("Profit obtained in each timestep:", profit_t)

# output_file = "output.txt"

# # Open the file in write mode
# with open(output_file, "w") as file:
#     file.write(f"Production Prices: {Price}\n")
#     file.write(f"Demand: {Demand}\n")
#     file.write(f"Pre-bought Emission Allowance: {Pre_All}\n")
#     file.write(f"New Conventional Capacity: {New_CC}\n")
#     file.write(f"The objective value is: {obj * 10 ** -6} mEuro\n")
#     file.write(f"Quantities produced by each conventional generator: {sum_QC}\n")
#     file.write(f"Bought emission allowance (during) for each conventional generator: {sum_All}\n")
#     file.write(f"Total demand by scenario: {np.sum(Demand, axis=0)}\n")
#     file.write(f"Average Price by scenario: {Average_Price_S}\n")
#     file.write(f"Average Price over Time: {Average_Price_t}\n")
#     file.write(f"Total conventional quantity produced by scenario: {np.sum(Q_C, axis=(0, 1))}\n")
#     file.write(f"Bought emission allowance for each scenario: {np.sum(All, axis=(0, 1))}\n")
#     file.write(f"Profit obtained in each timestep: {profit_t}\n")

# print(f"Data written to '{output_file}' successfully.")
 