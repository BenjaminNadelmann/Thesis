import numpy as np 
import gurobipy as gp
from gurobipy import GRB

#Parameters 

#This works for some reason. 


C_g=0.8 #Emission per MWh [ton/MWh]
e_gCap=10 #Emission cap for generator [ton]
P_CO2=55 #Price of CO2 per ton [Euros]
c_g=50 #Euros per MWh
i_g=39 #Euros per MW
q_gbar0=10 #MW
alpha_t=0.8 #Slope of the inverse demand function
beta_tw=170 #Intercept of the inverse demand functio
rho_gtw=0.8 #Capacity factor
phi=1#binnary
conjectural=-alpha_t*(1+phi) #conjectural variation
kappa=0.1 #Global required renewable penetration

#Variables
def resmodel(c_g,i_g,q_gbar,alpha_t,beta_tw,rho_gtw, kappa, conjectural):
    m=gp.Model('RES')

    #Variables 
    phat=m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="prices")
    q_gtw=m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="production supplied")
    p_tw=m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="price")
    q_gbar=m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="new capacity")
    d_tw=m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="demand")
    eta_gtw=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="dual 1")
    ptylda=m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="dual ")
    epsilon_gtw=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="dual 2")
    E_gtw=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="dual 3")
    
    b1=m.addVar(vtype=GRB.BINARY, name="auxiliary variable 1")
    b2=m.addVar(vtype=GRB.BINARY, name="auxiliary variable 2")
    b3=m.addVar(vtype=GRB.BINARY, name="auxiliary variable 3")
    b4=m.addVar(vtype=GRB.BINARY, name="auxiliary variable 4")

    M=10000

    #constraints
    m.addConstr(epsilon_gtw - C_g*q_gtw == 0)
    m.addConstr(q_gtw==d_tw, name="supply equals demand")
    m.addConstr(q_gtw >= kappa * d_tw, name="renewable penetration")
    m.addConstr(phat == p_tw)
    m.addConstr(ptylda == beta_tw - alpha_t * d_tw)
    m.addConstr(p_tw == ptylda)
    
    m.addConstr(phat + conjectural * q_gtw - c_g + eta_gtw >= 0)
    m.addConstr(q_gtw >= 0)
    m.addConstr(phat + conjectural * q_gtw - c_g + eta_gtw <= M * (1-b1))
    m.addConstr(q_gtw <= M * b1)

    m.addConstr(i_g - rho_gtw * eta_gtw >= 0  )
    m.addConstr(q_gbar >= 0) 
    m.addConstr(i_g - rho_gtw * eta_gtw <= M * b2)
    m.addConstr(q_gbar <= M * (1-b2))

    m.addConstr(rho_gtw * (q_gbar0 + q_gbar)-q_gtw >= 0)
    m.addConstr(eta_gtw >= 0)
    m.addConstr(rho_gtw * (q_gbar0 + q_gbar)-q_gtw <= M * b3)
    m.addConstr(eta_gtw <= M * (1-b3))

    m.addConstr(epsilon_gtw >= 0)
    m.addConstr(E_gtw >= 0)
    m.addConstr(epsilon_gtw <= M * b4)
    m.addConstr(E_gtw <= M * (1-b4))
    
    m.setObjective(beta_tw*d_tw - (1/2)*alpha_t*d_tw**2 - (c_g*q_gtw - P_CO2 * (e_gCap - epsilon_gtw)+ i_g*q_gbar), GRB.MAXIMIZE)

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
       print("quantity =", q_gtw.x)
       print("demand =", d_tw.x)
       print("price =", p_tw.x)
       
       print("new capacity", q_gbar.x)
       print("emission", epsilon_gtw.x)
       print("dual emission", E_gtw.x)
       print("Dual investment", eta_gtw.x)

       
       
    

    return m
resmodel(c_g,i_g,q_gbar0,alpha_t,beta_tw,rho_gtw, kappa, conjectural)