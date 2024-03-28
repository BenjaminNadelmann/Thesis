#!/usr/bin/env python
# coding: utf-8

# In[86]:


import numpy as np 
import gurobipy as gp
from gurobipy import GRB
from linopy import Model
import pyomo.environ as pyo
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import random
from scipy.stats import multivariate_normal


def resmodel(rho_gtw, beta_tw, pi_w, C_g, i_g, rho_jtw, i_j, q_j0, alpha, P_CO2_tw, epsilon_max_g, p_CO2_max, q_g0, xi_gtw, T, G, J, W):
    
    m = pyo.ConcreteModel()

    
    # Index sets
    m.G = pyo.RangeSet(0, G - 1)
    m.J = pyo.RangeSet(0, J - 1)
    m.T = pyo.RangeSet(0, T - 1)
    m.W = pyo.RangeSet(0, W - 1)
    

    ### Primal Variables
    
    ## For Conventional generators
    
    m.q_gtw    = pyo.Var(m.G, m.T, m.W, domain = pyo.NonNegativeReals)
    m.p_tw     = pyo.Var(m.T, m.W,      domain = pyo.NonNegativeReals)
    m.d_tw     = pyo.Var(m.T, m.W,      domain = pyo.NonNegativeReals)
    m.E_gtw    = pyo.Var(m.G, m.T, m.W)
    m.q_gbar   = pyo.Var(m.G,           domain = pyo.NonNegativeReals) 
    m.E_g      = pyo.Var(m.G,           domain = pyo.NonNegativeReals)
    m.gamma_gw = pyo.Var(m.G, m.W,      domain = pyo.NonNegativeReals)   # Changed to gamma_gw was Phi_gw # CVaR axuxillary variable 
    m.sigm_g   = pyo.Var(m.G,           domain = pyo.NonNegativeReals)   # Equals var at the optimum
    m.CVaR_g   = pyo.Var(m.G)                                            # Calculating conditional value at risk for g genrator
    m.Ex_pr_g  = pyo.Var(m.G, m.T, m.W)                                  # To calculate expected profit
    
    ## For Renewable generators
    
    m.q_jtw    = pyo.Var(m.J, m.T, m.W, domain = pyo.NonNegativeReals)
    m.q_jbar   = pyo.Var(m.J,           domain = pyo.NonNegativeReals)
    m.gamma_jw = pyo.Var(m.J, m.W,      domain = pyo.NonNegativeReals)   # Changed to gamma_jw was Phi_jw 
    m.sigm_j   = pyo.Var(m.J,           domain = pyo.NonNegativeReals)
    m.CVaR_j   = pyo.Var(m.J)
    m.Ex_pr_j  = pyo.Var(m.J, m.T, m.W) # RES generator j expected profit
    
    ### Dual variables 
    # For conventional variables
    
    m.eta_gtw    = pyo.Var(m.G, m.T, m.W, domain = pyo.NonNegativeReals)
    m.nu_min_gtw = pyo.Var(m.G, m.T, m.W, domain = pyo.NonNegativeReals)
    m.nu_max_gtw = pyo.Var(m.G, m.T, m.W, domain = pyo.NonNegativeReals)
    m.zeta_gtw   = pyo.Var(m.G, m.T, m.W, domain = pyo.NonNegativeReals)    # Was changed from psi_gtw to zeta_gtw
    m.mu_gtw     = pyo.Var(m.G, m.T, m.W)                                   # Was changed from mu_gw to mu_gtw
    m.Delta_gw   = pyo.Var(m.G, m.W,      domain = pyo.NonNegativeReals)
    m.nu_g_min   = pyo.Var(m.G,           domain = pyo.NonNegativeReals)
    m.theta_gw   = pyo.Var(m.G, m.W,      domain = pyo.NonNegativeReals)
    m.phi_g      = pyo.Var(m.G,           domain = pyo.NonNegativeReals)    
    # m.nu_g_max   = pyo.Var(m.G,           domain  = pyo.NonNegativeReals)
    m.nu_g_cap   = pyo.Var(m.G,           domain = pyo.NonNegativeReals)
    
    # For Res generators
    
    m.eta_jtw   =   pyo.Var(m.J, m.T, m.W, domain = pyo.NonNegativeReals)
    m.Delta_jw  =   pyo.Var(m.J, m.W,      domain = pyo.NonNegativeReals)
    m.theta_jw  =   pyo.Var(m.J, m.W,      domain = pyo.NonNegativeReals)
    m.zeta_jtw  =   pyo.Var(m.J, m.T, m.W, domain = pyo.NonNegativeReals)   # Was changed from psi_jtw to zeta_jtw
    m.phi_j     =   pyo.Var(m.J,           domain = pyo.NonNegativeReals)   
    
    ### Binary variables
    # For conventional generators
    
    m.psi_bar_gtw =    pyo.Var(m.G, m.T, m.W, domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_g_gtw   =    pyo.Var(m.G, m.T, m.W, domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_max_gtw =    pyo.Var(m.G, m.T, m.W, domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_min_gtw =    pyo.Var(m.G, m.T, m.W, domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_c_gw    =    pyo.Var(m.G, m.W,      domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_fA_gtw  =    pyo.Var(m.G, m.T, m.W, domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_g_bar   =    pyo.Var(m.G,           domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_g_min   =    pyo.Var(m.G,           domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_g_cap   =    pyo.Var(m.G,           domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx

    # For renewable generators
    
    m.psi_bar_jtw =    pyo.Var(m.J, m.T, m.W, domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_j_jtw   =    pyo.Var(m.J, m.T, m.W, domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_c_jw    =    pyo.Var(m.J, m.W,      domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_fB_jtw  =    pyo.Var(m.J, m.T, m.W, domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    m.psi_j_bar   =    pyo.Var(m.J,           domain = pyo.Binary)  # Changed from phi_xxx to psi_xxx
    
    # Profit definitions
    m.profit_g    =    pyo.Var(m.G, m.T, m.W)
    m.profit_j    =    pyo.Var(m.J, m.T, m.W)
    

    

    M_p1 = 10000000
    M_p2 = 10000000
    M_p3 = 10000000
    M_p4 = 10000000
    M_p5 = 10000000
    M_p6 = 10000000
    M_p7 = 10000000
    M_p8 = 10000000
    M_p9 = 10000000
    
    
    M_d1 = 10000000 # Check the values
    M_d2 = 10000000
    M_d3 = 10000000
    M_d4 = 10000000
    M_d5 = 10000000
    M_d6 = 10000000
    M_d7 = 10000000
    M_d8 = 10000000
    M_d9 = 10000000
    
    


   #Constraints
    def constraint_rule_1(m, g, j, t, w):
        return pyo.quicksum(m.q_jtw[j, t, w] for j in m.J) + pyo.quicksum(m.q_gtw[g, t, w] for g in m.G) == m.d_tw[t, w]

    m.Constraint1 = pyo.Constraint(m.G, m.J, m.T, m.W, rule = constraint_rule_1)
    
    
    def constraint_rule_2(m, j, t, w):
        return pyo.quicksum(pyo.quicksum(m.q_jtw[j, t, w] for j in m.J) for t in m.T) >= kappa * pyo.quicksum( m.d_tw[t, w] for t in m.T)

    m.Constraint2 = pyo.Constraint( m.J, m.T, m.W, rule = constraint_rule_2)


    def constraint_rule_3(m, t, w):
        return m.p_tw[t, w] == beta_tw[t, w] - alpha * m.d_tw[t, w]

    m.Constraint3 = pyo.Constraint(m.T, m.W, rule = constraint_rule_3)
    
    #Forgot to multiply the xi_gtw with my_gw 
    def KKT_rule_4a(m, g, t, w):
        return (beta - (m.Delta_gw[g, w]/pi_w[w]) - 1) * pi_w[w] * (m.p_tw[t, w] - (alpha * (1 + Phi) * m.q_gtw[g, t, w]) - C_g[g]) \
               + m.eta_gtw[g, t, w] + (P_CO2_tw[t,w]*m.Delta_gw[g, w] - pi_w[w]*(beta-1)*P_CO2_tw[t,w] - m.mu_gtw[g, t, w]) * xi_gtw[g,t,w]  == 0 

    m.KKT_constraint_4a = pyo.Constraint(m.G, m.T, m.W, rule = KKT_rule_4a)
    

    #First of all it is 11 a not 10a, however it is a cosmetic. Other thing, there is pi  missing with delta denominator 
    def KKT_rule_410a(m, j, t, w):
        return (beta - (m.Delta_gw[j, w]/pi_w[w]) - 1) * pi_w[w] * (m.p_tw[t, w] - alpha * (1 + Phi) * m.q_jtw[j, t, w] ) + \
               + m.eta_gtw[j, t, w] - m.zeta_gtw[j, t, w] == 0

    m.KKT_constraint_410a = pyo.Constraint(m.J, m.T, m.W, rule = KKT_rule_410a)


    def KKT_rule_4b(m, g, t, w):
        return  - (beta - m.Delta_gw[g, w] - 1) * i_g[g] - m.eta_gtw[g, t, w] * rho_gtw[g, t, w] - m.phi_g[g] == 0

    m.KKT_constraint_4b = pyo.Constraint(m.G, m.T, m.W, rule = KKT_rule_4b)
    
    def KKT_rule_10b(m, j, t, w):
        return  -(beta - m.Delta_jw[j, w] - 1) * i_j[j] - m.eta_jtw[j, t, w] * rho_jtw[j, t, w] - m.phi_j[j] == 0

    m.KKT_constraint_10b = pyo.Constraint(m.J, m.T, m.W, rule = KKT_rule_10b)
    

    def KKT_rule_4c(m, g, t, w):
        return m.mu_gtw[g, t, w] - m.nu_min_gtw[g, t, w] + m.nu_max_gtw[g, t, w] == 0

    m.KKT_constraint_4c = pyo.Constraint(m.G, m.T, m.W, rule = KKT_rule_4c)

    #This actually 4e not 4d
    def KKT_rule_4d(m, g, w):
        return ((beta * pi_w[w]) / (1 - v)) - m.Delta_gw[g, w] - m.theta_gw[g, w] == 0

    m.KKT_constraint_4d = pyo.Constraint(m.G, m.W, rule = KKT_rule_4d)
    
    #This is 4d actually 
    def KKT_rule_4Em(m, g, t, w):
        return (beta-1)* (pi_w[w] * P_CO2_tw[t, w] - p_CO2_max) +  m.Delta_gw[g,w]* (P_CO2_tw[t, w] - p_CO2_max) + 1/abs(T) * m.mu_gtw[g, t, w] - m.nu_g_min[g] + m.nu_g_cap[g] == 0

    m.KKT_constraint_4Em = pyo.Constraint(m.G, m.T, m.W, rule = KKT_rule_4Em)
    
    def KKT_rule_10c(m, j, w):
        return ((beta * pi_w[w]) / (1 - v)) - m.Delta_jw[j, w] - m.theta_jw[j, w] == 0

    m.KKT_constraint_10c = pyo.Constraint(m.J, m.W, rule = KKT_rule_10c)

    #This is 4f actually
    def KKT_rule_4e(m, g, w):
        return - beta + pyo.quicksum(m.Delta_gw[g, w] for w in m.W) == 0

    m.KKT_constraint_4e = pyo.Constraint(m.G, m.W, rule = KKT_rule_4e)
    
    def KKT_rule_10d(m, j, w):
        return -beta + pyo.quicksum(m.Delta_jw[j, w] for w in m.W) == 0

    m.KKT_constraint_10d = pyo.Constraint(m.J, m.W, rule = KKT_rule_10d)

    #actually 4g, added abs 
    def KKT_rule_4f(m, g, t, w):
        return xi_gtw[g, t, w] * m.q_gtw[g, t, w] - 1/abs(T) * m.E_g[g] - m.E_gtw[g, t, w]  == 0

    m.KKT_constraint_4f = pyo.Constraint(m.G, m.T, m.W, rule = KKT_rule_4f)

    def constraint_rule_6a(m, g, t, w):
        return (rho_gtw[g, t, w] * (q_g0[g] + m.q_gbar[g]) - m.q_gtw[g, t, w]) >= 0

    m.Constraint6a = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6a)
    
    #this is acutally 12a
    def constraint_rule_11a(m, j, t, w):
        return (rho_jtw[j, t, w] * (q_j0[j] + m.q_jbar[j]) - m.q_jtw[j, t, w]) >= 0

    m.Constraint11a = pyo.Constraint(m.J, m.T, m.W, rule = constraint_rule_11a)
    
    
    def constraint_rule_6b(m, g, t, w):
        return (rho_gtw[g, t, w] * (q_g0[g] + m.q_gbar[g]) - m.q_gtw[g, t, w]) <= (1 - m.psi_bar_gtw[g, t, w]) * M_p1

    m.Constraint6b = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6b)
    
    def constraint_rule_11b(m, j, t, w):
        return rho_jtw[j, t, w] * (q_j0[j] + m.q_jbar[j]) - m.q_jtw[j, t, w] <= (1 - m.psi_bar_jtw[j, t, w]) * M_p1

    m.Constraint11b = pyo.Constraint(m.J, m.T, m.W, rule = constraint_rule_11b)
    
    
    def constraint_rule_6c(m, g, t, w):
        return m.eta_gtw[g, t, w] <=  m.psi_bar_gtw[g, t, w] * M_d1 

    m.Constraint6c = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6c)
    
    def constraint_rule_11c(m, j, t, w):
        return m.eta_jtw[j, t, w] <=  m.psi_bar_jtw[j, t, w] * M_d1 

    m.Constraint11c = pyo.Constraint(m.J, m.T, m.W, rule = constraint_rule_11c)
    
    
    def constraint_rule_6d(m, g, t, w):
        return m.E_gtw[g, t, w] + epsilon_max_g[g] >= 0

    m.Constraint6d = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6d)
    
    def constraint_rule_6e(m, g, t, w):
        return m.E_gtw[g, t, w] + epsilon_max_g[g] <= (1- m.psi_min_gtw[g, t, w])*M_p2

    m.Constraint6e = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6e)
    
    def constraint_rule_6f(m, g, t, w):
        return m.nu_min_gtw[g, t, w] <= m.psi_min_gtw[g, t, w] * M_d2

    m.Constraint6f = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6f)
    
    def constraint_rule_6g(m, g, t, w):
        return epsilon_max_g[g] - m.E_gtw[g, t, w]  >= 0

    m.Constraint6g = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6g)
    
    def constraint_rule_6h(m, g, t, w):
        return epsilon_max_g[g] - m.E_gtw[g, t, w]  <= (1 - m.psi_max_gtw[g, t, w])*M_p3

    m.Constraint6h = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6h)
    
    def constraint_rule_6i(m, g, t, w):
        return m.nu_max_gtw[g, t, w] <= m.psi_max_gtw[g, t, w]*M_d3

    m.Constraint6i = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6i)

    def constrain_ruleq1(m,g,t,w):
        return m.q_gtw[g,t,w] >= 0 
    
    m.constrain_ruleq1=pyo.Constraint(m.G, m.T, m.W, rule = constrain_ruleq1)

    def constraint_rule_6k(m, g, t, w):
        return m.q_gtw[g, t, w] <= (1 - m.psi_g_gtw[g, t, w])*M_p4

    m.Constraint6k = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6k)
    
    def constraint_rule_11e(m, j, t, w):
        return m.q_jtw[j, t, w] <= (1 - m.psi_j_jtw[j, t, w])*M_p4

    m.Constraint11e = pyo.Constraint(m.J, m.T, m.W, rule = constraint_rule_11e)
    
    
    def constraint_rule_6l(m, g, t, w):
        return m.zeta_gtw[g, t, w] <= m.psi_g_gtw[g, t, w]*M_d4
           
    m.Constraint6l = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6l)
    
    #Need to clarify the Big M here
    def constraint_rule_11l(m, j, t, w):
        return m.zeta_jtw[j, t, w] <= m.psi_j_jtw[j, t, w]*M_d4
           
    m.Constraint11l = pyo.Constraint(m.J, m.T, m.W, rule = constraint_rule_11l)
    
    def constraint_rule_gbar(m, g,t,w):
        return m.q_gbar[g] >= 0
    
    m.constraint_rule_gbar = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_gbar)

    def constraint_rule_6n(m, g):
        return m.q_gbar[g] <= (1 - m.psi_g_bar[g])*M_p5

    m.Constraint6n = pyo.Constraint(m.G, rule = constraint_rule_6n)
    
    #that is actually 12h 
    def constraint_rule_11n(m, j):
        return m.q_jbar[j] <= (1 - m.psi_j_bar[j])*M_p5

    m.Constraint11n = pyo.Constraint(m.J, rule = constraint_rule_11n)
    
    
    def constraint_rule_6o(m, g):
        return m.phi_g[g] <=  m.psi_g_bar[g]*M_d5

    m.Constraint6o = pyo.Constraint(m.G, rule = constraint_rule_6o)
    
    #That is actually 12i
    def constraint_rule_11o(m, j):
        return m.phi_j[j] <=  m.psi_j_bar[j]*M_d5

    m.Constraint11o = pyo.Constraint(m.G, rule = constraint_rule_11o)
    
    #that is acutaly 6r
    def constraint_rule_6q(m, g, w):
        return m.gamma_gw[g, w] <= (1 - m.psi_c_gw[g,  w])*M_p6

    m.Constraint6q = pyo.Constraint(m.G, m.W, rule = constraint_rule_6q)
    
    #It is 12k 
    def constraint_rule_11q(m, j, w):
        return m.gamma_jw[j, w] <= (1 - m.psi_c_jw [j,  w])*M_p6

    m.Constraint11q = pyo.Constraint(m.J, m.W, rule = constraint_rule_11q)
    
    
    def constraint_rule_6r(m, g,  w):
        return m.theta_gw[g, w] <= m.psi_c_gw[g, w]*M_d6
            
    m.Constraint6r = pyo.Constraint(m.G, m.W, rule = constraint_rule_6r)
    #12 l 
    def constraint_rule_11r(m, j,  w):
        return m.theta_jw[j, w] <= m.psi_c_jw [j, w]*M_d6
            
    m.Constraint11r = pyo.Constraint(m.J, m.W, rule = constraint_rule_11r)
    
    
    def constraint_rule_6s(m, g, t, w):
        return m.gamma_gw[g, w] + pyo.quicksum(m.p_tw[t, w] * m.q_gtw[g, t, w] - C_g[g] * m.q_gtw[g, t, w] - i_g[g] * m.q_gbar[g] - m.E_g[g] * p_CO2_max + P_CO2_tw[t, w] * (m.E_g[g] - xi_gtw[g, t, w] * m.q_gtw[g, t, w]) for t in m.T) - m.sigm_g[g] >= 0

    m.Constraint6s = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6s)
    
    #this is 12m 
    def constraint_rule_11s(m, j, t, w):
        return m.gamma_jw[j, w] + pyo.quicksum(m.p_tw[t, w] * m.q_jtw[j, t, w] - i_j[j] * m.q_jbar[j]  for t in m.T) - m.sigm_j[j] >= 0

    #m.Constraint11s = pyo.Constraint(m.J, m.T, m.W, rule = constraint_rule_11s)
    
    
    
    def constraint_rule_6t(m, g, t, w):
        return m.gamma_gw[g, w] + pyo.quicksum(m.p_tw[t, w] * m.q_gtw[g, t, w] - C_g[g] * m.q_gtw[g, t, w] - i_g[g] * m.q_gbar[g] - m.E_g[g] * p_CO2_max + P_CO2_tw[t, w] * (m.E_g[g] - xi_gtw[g, t, w] * m.q_gtw[g, t, w]) for t in m.T) - m.sigm_g[g] <= (1 - m.psi_fA_gtw[g, t, w]) * M_p7

    m.Constraint6t = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6t)
    
    #this is 12n
    def constraint_rule_11t(m, j, t, w):
        return m.gamma_jw[j, w] + pyo.quicksum(m.p_tw[t, w] * m.q_jtw[j, t, w] - i_j[j] * m.q_jbar[j]  for t in m.T) - m.sigm_j[j] <= (1-m.psi_fB_jtw[j, t, w])*M_p7

    m.Constraint11t = pyo.Constraint(m.J, m.T, m.W, rule = constraint_rule_11t)
    
    # Profit definition
    def profit_g_def_rule(m, g, t, w):
        return m.profit_g[g, t, w] == m.p_tw[t, w] * m.q_gtw[g, t, w] -  C_g[g] * m.q_gtw[g, t, w] -i_g[g] * m.q_gbar[g] - m.E_g[g] * P_CO2_max + P_CO2_tw[t, w]* (m.E_g[g] - xi_gtw[g, t, w] * m.q_gtw[g, t, w])

    m.profit_g_def = pyo.Constraint(m.G, m.T, m.W, rule = profit_g_def_rule)

    def profit_j_def_rule(m, j, t, w):
        return m.profit_j[j, t, w] == m.p_tw[t, w] * m.q_jtw[j, t, w]  -i_j[j] * m.q_jbar[j] 

    m.profit_j_def = pyo.Constraint(m.J, m.T, m.W, rule = profit_j_def_rule)


    # Expected profit definition

    def expect_prof_g_rule(m, g, t, w):
        return m.Ex_pr_g[g, t, w] == pyo.quicksum(pi_w[w] *pyo.quicksum(m.profit_g[g, t, w] for t in m.T) for w in m.W)
    m.expect_prof_g = pyo.Constraint(m.G, m.T, m.W, rule = expect_prof_g_rule)

    def expect_prof_j_rule(m, j, t, w):
        return m.Ex_pr_j[j, t, w] == pyo.quicksum(pi_w[w] *pyo.quicksum(m.profit_j[j, t, w] for t in m.T) for w in m.W)
    m.expect_prof_j = pyo.Constraint(m.J, m.T, m.W, rule=expect_prof_j_rule)

    # CVaR constraints for groups g and j
    #Most likley m.phi_g should be replace by the sigma g and that is what we did 
    def cvar_g_def_rule(m, g, w):
        return m.CVaR_g[g] == m.sigm_g[g] - (1/(1 - v)) * pyo.quicksum(pi_w[w] * m.gamma_gw[g,w] for w in m.W)
    m.cvar_g_def = pyo.Constraint(m.G, m.W, rule = cvar_g_def_rule)

    #Most likley m.phi_j should be replace by the sigma j and that is what we did
    def cvar_j_def_rule(m, j, w):
        return m.CVaR_j[j] == m.sigm_j[j] - (1/(1 - v)) * pyo.quicksum(pi_w[w] * m.gamma_jw[j,w] for w in m.W)
    m.cvar_j_def = pyo.Constraint(m.J, m.W, rule = cvar_j_def_rule)

    
    def constraint_rule_6u(m, g, t, w):
        return m.Delta_gw[g, w] <= m.psi_fA_gtw[g, t, w]*M_d7

    m.Constraint6u = pyo.Constraint(m.G, m.T, m.W, rule = constraint_rule_6u)
    
    def constraint_rule_11u(m, j, t, w):
        return m.Delta_jw[j, w] <= m.psi_fB_jtw[j, t, w]*M_d7

    m.Constraint11u = pyo.Constraint(m.J,m.T, m.W, rule = constraint_rule_11u)
    
    def constraint_rule_7a(m, g):
        return m.E_g[g] <= (1 - m.psi_g_min[g]) * M_p8

    m.Constraint7a = pyo.Constraint(m.G, rule = constraint_rule_7a)
    
    def constraint_rule_7b(m, g):
        return m.nu_g_min[g] <= m.psi_g_min[g] * M_d8

    m.Constraint7b = pyo.Constraint(m.G, rule = constraint_rule_7b)
    #The equality has wrong direction in paper 7d 
    def constraint_rule_7c(m, g):
        return epsilon_cap_g[g] - m.E_g[g] >= 0

    m.Constraint7c = pyo.Constraint(m.G, rule = constraint_rule_7c)
    
    
    def constraint_rule_7d(m, g):
        return epsilon_cap_g[g] - m.E_g[g] <= (1 - m.psi_g_cap[g]) * M_p9

    m.Constraint7d = pyo.Constraint(m.G, rule = constraint_rule_7d)
    
    def constraint_rule_7e(m, g):
        return m.nu_g_cap[g] <= m.psi_g_cap[g] * M_d9

    m.Constraint7e = pyo.Constraint(m.G, rule = constraint_rule_7e)
    
    
    def objective_function(m):
        return (pyo.quicksum(pi_w[w] * (beta_tw[t, w] * m.d_tw[t, w] - 0.5 * alpha * m.d_tw[t, w]*m.d_tw[t,w]) for t in m.T for w in m.W)
            + pyo.quicksum(pi_w[w] * P_CO2_tw[t, w] * (m.E_g[g] - xi_gtw[g, t, w] * m.q_gtw[g, t, w]) for g in m.G for t in m.T for w in m.W)
            - pyo.quicksum(pi_w[w] * C_g[g] * m.q_gtw[g, t, w] for g in m.G for t in m.T for w in m.W)
            - pyo.quicksum(i_g[g] * m.q_gbar[g] + m.E_g[g] * p_CO2_max for g in m.G)) - pyo.quicksum(i_j[j] * m.q_jbar[j]  for j in m.J)
                                                                                                                               
                                                                                                                               

    m.Objective = pyo.Objective(rule = objective_function, sense = pyo.maximize)

    

    solver = pyo.SolverFactory("gurobi")
    results = solver.solve(m, options={"NonConvex": 2})
    #log_infeasible_constraints(m)


#     # check if model solved to optimality
#     if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
#         print("Model is infeasible")
#         return
#     else:
# #         # check for complementary slackness
#         for g in m.G:
#             for t in m.T:
#                 for w in m.W:
# #                     print(f"m.q_gbar[{g}].value:", m.q_gbar[g].value)

#                     temp1 = m.eta_gtw[g, t, w].value * (rho_gtw[g, t, w] * (q_g0[g] + m.q_gbar[g].value) - m.q_gtw[g, t, w].value)
#                     #print(f"First condition: {temp1}")
#                     if temp1 !=0:
#                         print("Complementary slackness 1 not satisfied")

#                     temp2 = m.nu_min_gtw[g, t, w].value *(m.E_gtw[g, t, w].value + epsilon_max_g[g])
#                     if temp2 != 0:
#                         print("Complementary slackness  2 not satisfied ")

#                     temp3 = m.nu_max_gtw[g, t, w].value *(epsilon_max_g[g] - m.E_gtw[g, t, w].value)
#                     if temp3 != 0:
#                         print("Complementary slackness  3 not satisfied ")

#                     temp4 = m.q_gbar[g].value *m.phi_g[g].value
#                     if temp4 != 0:
#                         print("Complementary slackness  4 not satisfied ")

#                     temp5 = m.q_gtw[g, t, w].value * m.zeta_gtw[g, t, w].value
#                     if temp5 != 0:
#                         print("Complementary slackness  5 not satisfied ")

#                     temp6 = m.gamma_gw[g, w].value * m.theta_gw[g, w].value
#                     if temp6 != 0:
#                         print("Complementary slackness  6 not satisfied ")

#                     temp7 = m.Delta_gw[g, w].value * (m.gamma_gw[g, w].value + pyo.quicksum(m.p_tw[t, w].value * m.q_gtw[g, t, w].value - C_g[g] * m.q_gtw[g, t, w].value - i_g * m.q_gbar[g].value - epsilon_max_g[g] * p_CO2_max + P_CO2_tw[t, w] * (epsilon_max_g[g] - xi_gtw[g, t, w] * m.q_gtw[g, t, w].value) for t in m.T) - m.sigm_g[g])
#                     if temp7 != 0:

#                         print("Complementary slackness  7 not satisfied ")
            
            
#         for j in m.J:
#             for t in m.T:
#                 for w in m.W:

#                     tempr1 = m.eta_jtw[j, t, w].value * (rho_gtw[j, t, w] * (q_j0[j] + m.q_gbar[j].value) - m.q_gtw[j, t, w].value)
#                     if tempr1 !=0:
#                         print("Complementary slackness r1 not satisfied")

#                     tempr2 = m.q_jbar[j].value *m.phi_j[j].value
#                     if tempr2 != 0:
#                         print("Complementary slackness  r2 not satisfied ")

#                     tempr3 = m.q_jtw[j, t, w].value * m.zeta_jtw[j, t, w].value
#                     if tempr3 != 0:
#                         print("Complementary slackness  r3 not satisfied ")

#                     tempr4 = m.gamma_jw[j, w].value * m.theta_jw[j, w].value
#                     if tempr4 != 0:
#                         print("Complementary slackness  r4 not satisfied ")

#                     tempr5 = m.Delta_jw[g, w].value * (m.gamma_jw[j, w].value + pyo.quicksum(m.p_tw[t, w].value * m.q_jtw[j, t, w].value  - i_j[j] * m.q_jbar[j].value for t in m.T) - m.sigm_j[j])
#                     if tempr5 != 0:

#                         print("Complementary slackness  r5 not satisfied ")


    # return decision variables as arrays
    q_g_values      = np.zeros((G, T, W)) 
    E_values        = np.zeros((G, T, W)) 
    Phi_g_values    = np.zeros((G, W))
    p_values        = np.zeros((T, W)) 
    d_values        = np.zeros((T, W))
    q_gbar_values   = np.zeros((G))
    sigm_g_values   = np.zeros((G))
    Emission_values = np.zeros((G))

    for g in m.G:
        
        q_gbar_values[g] = m.q_gbar[g].value
        sigm_g_values[g] = m.sigm_g[g].value
        Emission_values[g] = m.E_g[g].value
        
        for t in m.T:
            for w in m.W:

                q_g_values[g, t, w] = m.q_gtw[g, t, w].value
                E_values[g, t, w]   = m.E_gtw[g, t, w].value
                Phi_g_values[g, w]    = m.gamma_gw[g, w].value
                p_values[t, w]      = m.p_tw[t, w].value
                d_values[t, w]      = m.d_tw[t, w].value
                

    q_j_values      = np.zeros((J, T, W)) 
    Phi_j_values    = np.zeros((J, W))
    q_jbar_values   = np.zeros((J))
    sigm_j_values   = np.zeros((J))

    for j in m.J:
        
        q_jbar_values[j] = m.q_jbar[j].value
        sigm_j_values[j] = m.sigm_j[j].value
        
        for t in m.T:
            for w in m.W:

                q_j_values[j, t, w]   = m.q_gtw[j, t, w].value
                Phi_j_values[j, w]    = m.gamma_jw[j, w].value

                
    return q_g_values, E_values, Emission_values, p_values, d_values, Phi_g_values, q_gbar_values, sigm_g_values, q_jbar_values, sigm_g_values, q_j_values, Phi_j_values


# In[87]:


np.random.seed(201)

G = 3
J = 3
T = 2
W = 2


rho_gtw_mean =  np.array([0.675, 0.705, 0.605])
rho_gtw_std = 0.002
rho_gtw_sigma = np.diag(rho_gtw_mean * rho_gtw_std)
rho_gtw = multivariate_normal(mean = rho_gtw_mean, cov = rho_gtw_sigma).rvs(size =T * W).reshape(G, T, W)

rho_jtw_mean =  np.array([0.391, 0.20, 0.251])
rho_jtw_std = 0.002
rho_jtw_sigma = np.diag(rho_jtw_mean * rho_gtw_std)
rho_jtw = multivariate_normal(mean = rho_jtw_mean, cov = rho_jtw_sigma).rvs(size = T * W).reshape(J, T, W)


xi_gtw_mean =  np.array([0.291, 0.302, 0.271])
xi_gtw_std = 0.002
xi_gtw_sigma = np.diag(xi_gtw_mean * xi_gtw_std)
xi_gtw = multivariate_normal(mean = xi_gtw_mean, cov = xi_gtw_sigma).rvs(size =T * W).reshape(G, T, W)

epsilon_max_g = np.array([10000, 9500, 12000])
epsilon_cap_g = np.array([10000, 9500, 12000])


q_g0 = np.array([2219, 2150, 5070])
q_j0 = np.array([2219, 2150, 5070])
#{}
P_CO2_max = 100
P_CO2_tw_mean = np.array([180])
P_CO2_tw_std = 0.0150
P_CO2_tw_sigma = np.diag(P_CO2_tw_mean * P_CO2_tw_std)
P_CO2_tw = multivariate_normal(mean = P_CO2_tw_mean, cov = P_CO2_tw_sigma).rvs(size =T * W).reshape(T, W)

beta_tw_mean = np.array([180])
beta_tw_std = 0.0150
beta_tw_sigma = np.diag(beta_tw_mean * beta_tw_std)
beta_tw = multivariate_normal(mean = beta_tw_mean, cov = beta_tw_sigma).rvs(size =T * W).reshape(T, W)

C_g = np.array([14.5, 31, 22]) #Reasonalbe costs 
i_g = np.array([425, 170, 688])
i_j = np.array([405, 250, 500])

pi_w = (1/W) * np.ones(W)
alpha = 0.15 #This maybe should be various numbers
beta = 0         #Risk neutral it means now I think          # 0 or vector of numbers between 0 and 1 as CVaR parameters
v = 0                   # Confidence interval

kappa = 0               # Global required renewable penetration

Phi = -1                   # or -1 which is a reaction parameter



# In[88]:


q_g_values, E_values, Emission_values, p_values, d_values, Phi_g_values, q_gbar_values, sigm_g_values, q_jbar_values, sigm_g_values, q_j_values, Phi_j_values = resmodel(rho_gtw, beta_tw, pi_w, C_g, i_g, rho_jtw, i_j, q_j0, alpha, P_CO2_tw, epsilon_max_g, P_CO2_max, q_g0, xi_gtw, T, G, J, W)

q_g_values, E_values, Emission_values, p_values, d_values, Phi_g_values, q_gbar_values, sigm_g_values, q_jbar_values, sigm_g_values, q_j_values, Phi_j_values


# In[ ]:




