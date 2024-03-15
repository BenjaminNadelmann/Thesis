
import numpy as np
from gurobipy import GRB
import gurobipy as gp

T = 5 #Number of timesteps
W = 5 #Number of scenarios
GC = 3 #Number of Conventional Generators 
G = 5 #Number of generators

pi_w = {i: 1/W for i in range(W)}
#print(pi_w)
tau_t = {i: 1 for i in range(T)}
#print(tau_tw)
v_g = {i: 1/G for i in range(G)}
#print(v_g)
beta_w = [50, 75, 100, 125, 150]
#print(beta_tw)
c_g = [10, 10, 10, 0, 0]
i_g = [50000, 50000, 50000, 100000, 100000]
alpha_t = {i: 1/T for i in range(T)}
zeta = 0.2
rho = 0.6
q_g0 = [100, 100, 100, 100, 100] 
kappa = 0.4 
P_CO2 = 55 
ecap = 100 
conjectural_variation = 1 #Not sure if these parameters are actually parameters
beta = 0 #Not sure if these parameters are actually parameters


def MIQP_Sets(pi_w, tau_t, v_g, beta_w, alpha_t, c_g, i_g, zeta, rho, q_g0, kappa, P_CO2, ecap, conjectural_variation, beta):

    m=gp.Model('Social_Welfare_Model')
    
    #Defining the primal variables:
    q_gtw=m.addVars(G, T, W, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="quantity")
    q_gbar=m.addVars(G, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="New_Capacity")
    epsilon_gtw=m.addVars(G, T, W, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="emission") 
    d_tw=m.addVars(T, W, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="demand")
    R_gtw=m.addVars(GC, T, W, lb=0,vtype=GRB.CONTINUOUS, name="revenue")
    C_gtw=m.addVars(GC, T, W, lb=0,vtype=GRB.CONTINUOUS, name="cost")
    sigma_g=m.addVars(G, vtype=GRB.BINARY, name="Auxiliary_Variable1")
    psi_gw=m.addVars(G, W, lb=0, vtype=GRB.BINARY, name="Auxiliary_Variable2")

    #Defining the Dual Variables:
    p_tw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-1")
    eta_gtw=m.addVars(G, T, W, lb=0,vtype=GRB.CONTINUOUS, name="Dual-2")
    #underline_gamma_gtw=m.addVars(G, T, W, lb=0,vtype=GRB.CONTINUOUS, name="Dual-3") 
    #Bar_gamma_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-4")
    phi_g=m.addVars(G, lb=0, vtype=GRB.CONTINUOUS, name="Dual-5")
    delta_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-6")
    Theta_gw=m.addVars(G, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-7")
    muC_gtw=m.addVars(GC, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-8")
    muR_gtw=m.addVars(GC, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-9")
    l_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-10")
    psi_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-11")
    xi_gtw=m.addVars(T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-12")
    vR_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-13")
    vC_gtw=m.addVars(G, T, W, lb=0, vtype=GRB.CONTINUOUS, name="Dual-14")

    #The generators' profit optimization model (Maybe it should also be indexed through the different sets)
    for g,t,w in range(G,T,W):
        Z_gtw = pi_w[w] * tau_t[t] * (p_tw[t,w] * q_gtw[g,t,w] - (c_g[g] * q_gtw[g,t,w] + P_CO2*(R_gtw[g,t,w] - C_gtw[g,t,w])))-i_g[g] * q_gbar[g]

    #Defining constraints Upper Level:

    for g,t,w in range(G,T,W):
        m.addConstr(q_gtw[g,t,w] == d_tw[t,w], name="UL-1")
        m.addConstr(tau_t[t] * q_gtw[g,t,w] >= kappa * tau_t[t] * d_tw[t,w], name="UL-2")
        m.addConstr(p_tw[t,w] == beta_w[w] - alpha_t[t]*d_tw[t,w], name="UL-3")

    #Defining constraints Lower Level: Equality 

    for g,t,w in range(G,T,W):
        m.addConstr(((1 - beta-delta_gtw[g,t,w]) * pi_w[w] * tau_t[t] * (p_tw[t,w] + (-alpha_t[t] * (1+conjectural_variation) * q_gtw[g,t,w]) - c_g[g])
                    + eta_gtw,[g,t,w] + zeta * xi_gtw[t,w] - psi_gtw[g,t,w] == 0),name="1.7a")
        m.addConstr((-eta_gtw[g,t,w] * rho - i_g[g] * (1 - beta - delta_gtw[g,t,w]) - phi_g[g] == 0), name="1.7b")
        m.addConstr(vC_gtw[g,t,w] - vR_gtw[g,t,w] - xi_gtw[g,t,w] - l_gtw[g,t,w] == 0, name="1.7c")
        m.addConstr(- (1 - beta - delta_gtw[g,t,w]) * pi_w[w] * tau_t[t] * P_CO2 - muR_gtw[g,t,w] - vR_gtw[g,t,w] ==0, name="1.7d")
        m.addConstr((1 - beta - delta_gtw[g,t,w]) * pi_w[w] * tau_t[t] * P_CO2 - muC_gtw[g,t,w] - vC_gtw[g,t,w] == 0, name="1.7e")
        m.addConstr((beta * pi_w[w]) / (1 - v_g[g]) - delta_gtw[g,t,w] - Theta_gw[g,t,w] == 0, name="1.7f")
        m.addConstr( - beta + delta_gtw[g,t,w] == 0, name="1.7g")

    #Constraints that only target the conventional generators
    for gc in range(GC):
        for t in range(T): 
            for w in range(W):
                m.addConstr(epsilon_gtw[gc,t,w] - zeta * q_gtw[gc,t,w] == 0, name="1.7h")
                m.addConstr(R_gtw[gc,t,w] - (ecap- epsilon_gtw[gc,t,w]) == 0, name="1.7i")
                m.addConstr(C_gtw[gc,t,w] - (epsilon_gtw[gc,t,w] - ecap) == 0, name="1.7j")

    #Defining constraints Lower Level: complementarity 

    #This is done by linearizing the complementarity constraints through the Big M technique:
    
    C = 8 #Number of complementarity constraints 
    b = m.addVars(C, vtype=GRB.BINARY, name="BigM_AV")
    M1, M2 = 600, 1000

    #Big M constraints where all generators are involved 
    for g,t,w in range(G,T,W):
        m.addConstr(rho * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] >= 0, name="P1_BigM1")
        m.addConstr(rho * (q_g0[g] + q_gbar[g]) - q_gtw[g,t,w] <= M1 * (1-b[0]), name="P2_BigM1")
        m.addConstr(eta_gtw[g,t,w] <= M1 * b[0], name="D_BigM1")
        m.addConstr(q_gbar[g] <= M1 * (1-b[1]), name="P_BigM2")
        m.addConstr(phi_g[g] <= M1 * b[1], name="D_BigM2")
        m.addConstr(q_gtw[g,t,w] <= M1 * (1-b[2]), name="P_BigM3")
        m.addConstr(psi_gtw[g,t,w] <= M1 * b[2], name="D_BigM3")
        m.addConstr(epsilon_gtw[g,t,w] <= M1 * (1-b[3]), name="P_BigM4")
        m.addConstr(l_gtw[g,t,w] <= M1 * b[3], name="D_BigM4")
        m.addConstr(psi_gw[g,w] <= M1 * (1-b[4]), name="P_BigM5")
        m.addConstr(Theta_gw[g,w] <= M1 * b[4], name="D_BigM5")
        m.addConstr(psi_gw[g,w] + Z_gtw[g,t,w] - sigma_g[g] >= 0, name="P1_BigM6")
        m.addConstr(psi_gw[g,w] + Z_gtw[g,t,w] - sigma_g[g] <= M2 * (1-b[5]), name="P2_BigM6")
        m.addConstr(delta_gtw[g,t,w] <= M2 * b[5], name="D_BigM6")
    

    #Big M constraints that only target the conventional generators
    for gc in range(GC):
        for t in range(T): 
            for w in range(W):
                m.addConstr(R_gtw[gc,t,w] <= M1 * (1-b[6]), name="P_BigM7")
                m.addConstr(muR_gtw[gc,t,w] <= M1 * b[6], name="D_BigM7")
                m.addConstr(C_gtw[gc,t,w] <= M1 * (1-b[7]), name="P_BigM8")
                m.addConstr(muC_gtw[gc,t,w] <= M1 * b[7], name="D_BigM8")

    #Writing out the objective function
    for g,t,w in range(G,T,W):
        objective = pi_w[w] * tau_t[t] * (beta_w[w] * d_tw[t,w] - (1/2) * alpha_t[t] * d_tw[t,w]**2 - (c_g[g] * q_gtw[g,t,w] + P_CO2*(R_gtw[g,t,w] - C_gtw[g,t,w])))-i_g[g] * q_gbar[g]

    m.setObjective(objective, GRB.MAXIMIZE)

    #solving the model
    m.update()    
    m.optimize()

    return 

MIQP_Sets(pi_w, tau_t, v_g, beta_w, alpha_t, c_g, i_g, zeta, rho, q_g0, kappa, P_CO2, ecap, conjectural_variation, beta)