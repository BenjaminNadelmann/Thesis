
#Importing the pacakges
import numpy as np
from gurobipy import GRB
import gurobipy as gp

#Defining the model
m=gp.Model('Social_Welfare_Model')

#Defining the sets
#time = range(len(23)) #Hopefully makes up a time set ranging from [0..23]
#Scenario = range(len(50)) #Should make a scenario set ranging from [0..50]
#generator = range(len(10)) #Should make a generator set ranging from [0..10]


#Defining the variables, however a lot of the values are arbitrary. 
q_gtw=m.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="quantity")
qgbar=m.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="New_Capacity")
epsilon_gtw=m.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="emission") 
dtw=m.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="demand")
R_gtw=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="revenue")
C_gtw=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="cost")
sigma=m.addVar(vtype=GRB.BINARY, name="Auxiliary_Variable")
psi=m.addVar(vtype=GRB.BINARY, name="Auxiliary_Variable")
#beta=m.addVar(vtype=GRB.BINARY, name="Auxiliary_Variable")

#Defining the Dual variables, NOT SURE IF THIS IS CORRECT!!, If not then we'll change it later:
p_tw=m.addVar(vtype=GRB.CONTINUOUS, name="Dual variable for the price")
eta_gtw=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="Dual variable for the quantity produced")
underline_gamma_gtw=m.addVar(ub=0,vtype=GRB.CONTINUOUS, name="Dual variable for the ramping limits") #ub=0, maybe this should be added though it's a ramping down (so maybe it's negative)
Bar_gamma_gtw=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="Dual variable for the ramping limits")
lambda_w=m.addVar(vtype=GRB.CONTINUOUS, name="Dual variable for renewab")
phi_g=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="Dual variable for the new capacity in generator")
delta_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="daul variable for the profit function ")
Theta_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="daul variable for the auxilary variable")
muC_gtw=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="daul variable for revenue")
muR_gtw=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="daul variable for cost ")
l_gtw=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="daul variable for emissions")
psi_gtw=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="daul variable for the quantity")
xi_tw=m.addVar(vtype=GRB.CONTINUOUS, name="daul variable for the generator intensity")
vR_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="daul variable for the revenue emission cap")
vC_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="daul variable for the cost emission cap")



#Defining the parameters - Prices are in Euro 

pi_w=1 #Probability, for now it is 1 just to test one case   
tau_tw=1 #Time, for now it is 1 just to test one case
v_g=0 #This is confidence level for the generator, for now it is 0 just to test one case
beta_tw=155 #intercept of demand function 
alpha=0.5 #slope of demand function
c_g=10 #Marginal cost of production
i_g=10000 #Cost of new generating technology 
#r_gdown = 0 #rump down  - not suere if we are going to use that 
#r_gup = 0 #rump up - not suere if we are going to use that
zeta_g = 0.2 #emission intensity 
rho_gtw = 0.6 #capacity facto 
qg0r = 100 #Initial capacity of the renewable generator
qg0c = 100 #Initial capacity of the conventioanl generator
kappa = 0.4 #global required renewable penetration 
PCO2=55 #Price of CO2 per ton
ecap_g=100 #Emission cap for generator
conjectural_variation = 1 # this can be either 0 or 1, for now it is 0 
beta = 0 # risk aversion (should probably be a binary variable)

#The generators' profit optimization model
Z_gtw = pi_w*tau_tw*(p_tw*q_gtw-(c_g*q_gtw+PCO2*(R_gtw-C_gtw)))-i_g*qgbar



#Defining the SOS constraints  https://www.gurobi.com/documentation/current/refman/py_model_addsos.html
#https://www.gurobi.com/documentation/current/refman/sos_constraints.html#subsubsection:SOSConstraints

#Constraints  normal ones: 
#gp.quicksum seems to be the function used to make summations over an array of values and such
m.addConstr(q_gtw == dtw, name="Supply equals demand")
m.addConstr(tau_tw*q_gtw >= kappa * tau_tw * dtw, name="Renewable penetration")
m.addConstr(p_tw==beta-alpha*dtw, name="Price as a function of inverse demand")

#Now KKT Constraints (starting with the equality constraints)

m.addConstr((-(1-beta)*pi_w*tau_tw*delta_gtw*(c_g-p_tw-(-alpha*(1+conjectural_variation)*q_gtw))-underline_gamma_gtw+Bar_gamma_gtw+eta_gtw+zeta_g*xi_tw-psi_gtw==0),name="1.7a")

#In constraint 1.7a we haven't included the gamma_1+gtw and such yet, needs to be included when we write the constraints up with the different sets included. so that we can
#actually say "gamma of [t+1] or something like that"

m.addConstr((-eta_gtw*rho_gtw+i_g*(delta_gtw + beta -1)-phi_g==0), name="1.7b")
m.addConstr(-(vR_gtw-vC_gtw)- lambda_w - l_gtw == 0, name="1.7c")
m.addConstr((1-beta)*pi_w*tau_tw*delta_gtw*PCO2-muR_gtw-vR_gtw==0, name="1.7d")
m.addConstr(-(1-beta)*pi_w*tau_tw*delta_gtw*PCO2-muC_gtw-vC_gtw==0, name="1.7e")
m.addConstr((beta*pi_w)/(-1+v_g)-delta_gtw-Theta_gtw==0, name="1.7f")
m.addConstr(beta+delta_gtw==0, name="1.7g")
m.addConstr(epsilon_gtw-zeta_g*q_gtw == 0, name="1.7h")
m.addConstr(R_gtw - (ecap_g - epsilon_gtw) == 0, name="1.7i")
m.addConstr(C_gtw - (epsilon_gtw - ecap_g) == 0, name="1.7j")



#KKT SOS complementarity constraints
#First I will define the complementarity constraints and then try to use SOS 
#I did not implement constrains that had ramp down/up or the ones that are with t-1 

complementary_condition_1= ((rho_gtw*(qg0c+qgbar)-q_gtw),eta_gtw) #Does Not work 
complementary_condition_2= (qgbar,phi_g) #this means that either qbar or phi_g can take a non zero value at a time
complementary_condition_3=(q_gtw,psi_gtw)
complementary_condition_4=(epsilon_gtw,l_gtw)
complementary_condition_5=(R_gtw,muR_gtw)
complementary_condition_6=(C_gtw,muC_gtw)
complementary_condition_7=(psi_gtw,Theta_gtw)
complementary_condition_8=(psi_gtw+Z_gtw-sigma, delta_gtw) # does not work 


#Adding the SOS constraints - is it correct? I do not know 
m.addSOS(GRB.SOS_TYPE1, complementary_condition_1) #Does not work
m.addSOS(GRB.SOS_TYPE1, complementary_condition_2)
m.addSOS(GRB.SOS_TYPE1, complementary_condition_3)
m.addSOS(GRB.SOS_TYPE1, complementary_condition_4)
m.addSOS(GRB.SOS_TYPE1, complementary_condition_5)
m.addSOS(GRB.SOS_TYPE1, complementary_condition_6)
m.addSOS(GRB.SOS_TYPE1, complementary_condition_7)
m.addSOS(GRB.SOS_TYPE1, complementary_condition_8) #Does not work


#Objective Function
m.setObjective(pi_w*tau_tw*(beta_tw*dtw-(1/2)*alpha*dtw**2-(c_g*q_gtw+PCO2*(R_gtw-C_gtw)))-i_g*qgbar, GRB.MAXIMIZE)


#Running the optimization model:
m.optimize()