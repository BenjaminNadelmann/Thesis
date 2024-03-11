

#Importing the pacakges
import numpy as np
from gurobipy import GRB
import gurobipy as gp

#Defining the model
m=gp.Model('Social_Welfare_Model')

nTime = 5
nScenario = 5
nRenGen = 2
nConGen = 3
nGenerator = 5

#Defining the sets: Maybe not necessary 
# time = set(range(ntime)) #Hopefully makes up a time set ranging from [0..23]
# Scenario = set(range(nscenario)) #Should make a scenario set ranging from [0..30]
# Rengenerator = set(range(nRenGen)) #Should make a renewable generator set ranging from [0..5]
# Congenerator = set(range(nConGen)) #Should make a coneventional generator set ranging from [0..5]

# print(time)
# print(Scenario)
# print(Rengenerator)
# print(Congenerator)

#Defining the parameters - Prices are in Euro 

pi_w = {i: 1/nScenario for i in range(nScenario)}
#print(pi_w)
tau_tw = {i: 1 for i in range(nTime)}
#print(tau_tw)
v_g = {i: 1/nGenerator for i in range(nGenerator)}
#print(v_g)
beta_tw = [50, 75, 100, 125, 150]
#print(beta_tw)
c_g = 10
i_gR = 100000
i_gC = 50000
alpha = {i: 1/nGenerator for i in range(nGenerator)}
zeta_g = 0.2
rho_gtw = 0.6
q_g0r = 100 
q_g0c = 100 
kappa = 0.4 
P_CO2 = 55 
ecap_g = 100 
conjectural_variation = 1 #Not sure if these parameters are actually parameters
beta = 0 #Not sure if these parameters are actually parameters

# #Defining the variables, however a lot of the values are arbitrary.

q_gtw=m.addVars(nGenerator, nTime, nScenario, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="quantity")
print(q_gtw)
q_gbar=m.addVars(nGenerator, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="New_Capacity")
epsilon_gtw=m.addVars(nGenerator, nTime, nScenario, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="emission") 
d_tw=m.addVars(nTime, nScenario, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="demand")
R_gtw=m.addVars(nConGen, nTime, nScenario, lb=0,vtype=GRB.CONTINUOUS, name="revenue")
C_gtw=m.addVars(nConGen, nTime, nScenario, lb=0,vtype=GRB.CONTINUOUS, name="cost")
sigma_g=m.addVars(nGenerator, vtype=GRB.BINARY, name="Auxiliary_Variable")
psi_gw=m.addVars(nGenerator, nScenario, vtype=GRB.BINARY, name="Auxiliary_Variable")
#beta=m.addVar(vtype=GRB.BINARY, name="Auxiliary_Variable") #Not sure if this variable is a variable or parameter

m.update()
print(q_gtw)

#Defining the Dual variables:

p_tw=m.addVars(nTime, nScenario, vtype=GRB.CONTINUOUS, name="Dual variable for the price")
eta_gtw=m.addVars(nGenerator, nTime, nScenario, lb=0,vtype=GRB.CONTINUOUS, name="Dual variable for the quantity produced")
underline_gamma_gtw=m.addVars(nGenerator, nTime, nScenario, ub=0,vtype=GRB.CONTINUOUS, name="Dual variable for the ramping limits") #ub=0, maybe this should be added as it's a ramping down (so maybe it's negative)
Bar_gamma_gtw=m.addVars(nGenerator, nTime, nScenario, lb=0,vtype=GRB.CONTINUOUS, name="Dual variable for the ramping limits")
lambda_w=m.addVars(nScenario, vtype=GRB.CONTINUOUS, name="Dual variable for renewab")
phi_g=m.addVars(nGenerator, lb=0,vtype=GRB.CONTINUOUS, name="Dual variable for the new capacity in generator")
delta_gtw=m.addVars(nGenerator, nTime, nScenario, vtype=GRB.CONTINUOUS, name="daul variable for the profit function ")
Theta_gtw=m.addVars(nGenerator, nTime, nScenario, vtype=GRB.CONTINUOUS, name="daul variable for the auxilary variable")
muC_gtw=m.addVars(nConGen, nTime, nScenario, lb=0,vtype=GRB.CONTINUOUS, name="daul variable for revenue")
muR_gtw=m.addVars(nConGen, nTime, nScenario, lb=0,vtype=GRB.CONTINUOUS, name="daul variable for cost ")
l_gtw=m.addVars(nGenerator, nTime, nScenario, lb=0,vtype=GRB.CONTINUOUS, name="daul variable for emissions")
psi_gtw=m.addVars(nGenerator, nTime, nScenario, lb=0,vtype=GRB.CONTINUOUS, name="daul variable for the quantity")
xi_tw=m.addVars(nTime, nScenario, vtype=GRB.CONTINUOUS, name="daul variable for the generator intensity")
vR_gtw=m.addVars(nGenerator, nTime, nScenario, vtype=GRB.CONTINUOUS, name="daul variable for the revenue emission cap")
vC_gtw=m.addVars(nGenerator, nTime, nScenario, vtype=GRB.CONTINUOUS, name="daul variable for the cost emission cap")

m.update()
