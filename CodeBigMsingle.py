#Importing the necessary libraries 
import numpy as np
from gurobipy import GRB
import gurobipy as gp


#Defining the parameters - Prices are in Euro 
pi_w=1 #Probability, for now it is 1 just to test one case   
tau_tw=1 #Time, for now it is 1 just to test one case
v_g=0 #This is confidence level for the generator, for now it is 0 just to test one case
beta_tw=155 #intercept of demand function 
alpha_t=0.5 #slope of demand function
c_g=10 #Marginal cost of production
i_g=10000 #Cost of new generating technology 
#r_gdown = 10 #rump down  - not suere if we are going to use that 
#r_gup = 10 #rump up - not suere if we are going to use that
zeta_g = 0.2 #emission intensity 
rho_gtw = 0.6 #capacity facto 
qg0r = 100 #Initial capacity of the renewable generator
qg0c = 100 #Initial capacity of the conventioanl generator
kappa = 0.4 #global required renewable penetration 
PCO2=55 #Price of CO2 per ton
ecap_g=100 #Emission cap for generator
conjectural_variation = 1 # this can be either 0 or 1, for now it is 0 
beta = 1 # risk aversion (should probably be a binary variable)

def solve_MIQP_Problem(pi_w, tau_tw, v_g, beta_tw, alpha, c_g, i_g, zeta_g, rho_gtw, qg0r, qg0c, kappa, PCO2, ecap_g, conjectural_variation, beta):
    m=gp.Model('One case Model')
    
    #Defining the primal variables: 
    q_gtw=m.addVar(lb=0, ub=10000, vtype=GRB.CONTINUOUS, name="quantity")
    qgbar=m.addVar(lb=0, ub=10000, vtype=GRB.CONTINUOUS, name="New_Capacity")
    epsilon_gtw=m.addVar(lb=0, ub=10000, vtype=GRB.CONTINUOUS, name="emission") 
    d_tw=m.addVar(lb=0, ub=10000, vtype=GRB.CONTINUOUS, name="demand")
    R_gtw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="revenue")
    C_gtw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="cost")
    sigma=m.addVar(vtype=GRB.BINARY, name="Auxiliary_Variable")
    psi=m.addVar(vtype=GRB.BINARY, name="Auxiliary_Variable")
    
    #Defining the Dual variables: :
    p_tw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="Dual variable for the price")
    eta_gtw=m.addVar(lb=0, ub=10500, vtype=GRB.CONTINUOUS, name="Dual variable for the quantity produced")
    #underline_gamma_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="Dual variable for the ramping limits") #ub=0, maybe this should be added though it's a ramping down (so maybe it's negative)
    #bar_gamma_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="Dual variable for the ramping limits")
    
    phi_g=m.addVar(lb=0, ub=10000, vtype=GRB.CONTINUOUS, name="Dual variable for the new capacity in generator")
    delta_gtw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="daul variable for the profit function ")
    Theta_gw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="daul variable for the auxilary variable")
    muC_gtw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="daul variable for revenue")
    muR_gtw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="daul variable for cost ")
    l_gtw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="daul variable for emissions")
    psi_gtw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="daul variable for the quantity")
    xi_tw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="daul variable for the generator intensity")
    vR_gtw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="daul variable for the revenue emission cap")
    vC_gtw=m.addVar(lb=0, ub=10000,vtype=GRB.CONTINUOUS, name="daul variable for the cost emission cap")

    #The generators' profit optimization model
    Z_gtw = (p_tw*q_gtw-(c_g*q_gtw+PCO2*(R_gtw-C_gtw)))-i_g*qgbar
    
    
    #Defining the constraints
    m.addConstr(q_gtw == d_tw, name="Supply equals demand")
    m.addConstr(tau_tw*q_gtw >= kappa * tau_tw * d_tw, name="Renewable penetration")
    m.addConstr(p_tw== beta_tw- alpha_t* d_tw, name="Price as a function of inverse demand")
    #KKT equality constraints 
    m.addConstr(((1-beta-delta_gtw)*pi_w*tau_tw*(-c_g+p_tw+((-alpha)*(1+conjectural_variation)*q_gtw))+eta_gtw+zeta_g*xi_tw-psi_gtw==0),name="1.7a")
    m.addConstr((-eta_gtw*rho_gtw + i_g*(1-beta-delta_gtw)-phi_g==0), name="1.7b")
    m.addConstr(vC_gtw - vR_gtw - l_gtw - xi_tw == 0, name="1.7c")
    m.addConstr(-(1-beta-delta_gtw)*pi_w*tau_tw*PCO2 - muR_gtw - vR_gtw == 0 , name="1.7d")
    m.addConstr((1-beta-delta_gtw)*pi_w*tau_tw*PCO2 - muC_gtw - vC_gtw == 0 , name="1.7e")
    m.addConstr((beta * 1/(1-v_g)) - delta_gtw - Theta_gw == 0, name="1.7f")
    m.addConstr(-beta + delta_gtw == 0)
    m.addConstr(epsilon_gtw - zeta_g * q_gtw == 0)
    m.addConstr(R_gtw - ecap_g + epsilon_gtw == 0)
    m.addConstr(C_gtw + ecap_g - epsilon_gtw == 0)


    #Big M and the the binary variables 
    B1 = m.addVar(vtype=GRB.BINARY, name="BigM Auxiliary_Variable")
    B2 = m.addVar(vtype=GRB.BINARY, name="BigM Auxiliary_Variable")
    B3 = m.addVar(vtype=GRB.BINARY, name="BigM Auxiliary_Variable")
    B4 = m.addVar(vtype=GRB.BINARY, name="BigM Auxiliary_Variable")
    B5 = m.addVar(vtype=GRB.BINARY, name="BigM Auxiliary_Variable")
    B6 = m.addVar(vtype=GRB.BINARY, name="BigM Auxiliary_Variable")
    B7 = m.addVar(vtype=GRB.BINARY, name="BigM Auxiliary_Variable")
    B8 = m.addVar(vtype=GRB.BINARY, name="BigM Auxiliary_Variable")
    M1 = 600000
    M2 = 10000
    #Linearization of the complementary constraints using big M:

    m.addConstr(rho_gtw*(qg0c+qgbar)-q_gtw>= 0)
    m.addConstr(eta_gtw>= 0)
    m.addConstr(eta_gtw<= M1*B1)
    m.addConstr(rho_gtw*(qg0c+qgbar)-q_gtw<= M1*(1-B1))
    
    m.addConstr(qgbar >= 0)
    m.addConstr(phi_g >= 0)
    m.addConstr(phi_g <= M1*B2)
    m.addConstr(qgbar <= M1*(1-B2))
    
    m.addConstr(q_gtw >= 0)
    m.addConstr(psi_gtw >= 0)
    m.addConstr(psi_gtw <= M1*B3)
    m.addConstr(q_gtw <= M1*(1-B3))

    m.addConstr(epsilon_gtw >= 0)
    m.addConstr(l_gtw >= 0)
    m.addConstr(l_gtw <= M1*B4)
    m.addConstr(epsilon_gtw <= M1*(1-B4))

    m.addConstr(R_gtw >= 0)
    m.addConstr(muR_gtw >= 0)
    m.addConstr(muR_gtw <= M1*B5)
    m.addConstr(R_gtw <= M1*(1-B5))

    m.addConstr(C_gtw >= 0)
    m.addConstr(muC_gtw >= 0)
    m.addConstr(muC_gtw <= M1*B6)
    m.addConstr(C_gtw <= M1*(1-B6))

    m.addConstr(psi >= 0)
    m.addConstr(Theta_gw >= 0)
    m.addConstr(Theta_gw <= M1*B7)
    m.addConstr(psi <= M1*(1-B7))

    m.addConstr((psi+Z_gtw-sigma) >= 0)
    m.addConstr(delta_gtw >= 0)
    m.addConstr(delta_gtw <= M2*B8)
    m.addConstr((psi+Z_gtw-sigma) <= M2*(1-B8)) 


    #Objective function
    m.setObjective(pi_w*tau_tw*(beta_tw*d_tw-(1/2)*alpha_t*d_tw**2-(c_g*q_gtw+PCO2*(R_gtw-C_gtw)))-i_g*qgbar, GRB.MAXIMIZE)

    #Solving the model
    m.update()
    m.optimize()


    return m

solve_MIQP_Problem(pi_w, tau_tw, v_g, beta_tw, alpha_t, c_g, i_g, zeta_g, rho_gtw, qg0r, qg0c, kappa, PCO2, ecap_g, conjectural_variation, beta)