
#Importing the pacakges
import numpy as np
from gurobipy import * 
import gurobipy as grb

#Defining the model
m=Model('Social_Welfare_Model')

#Defining the variables, however a lot of the values are arbitrary. 

q_gtw=m.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="quantity")
qgbar=m.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="New_Capacity")
epsilon=m.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="emission") 
dtw=m.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="demand")
R=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="revenue")
C=m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="cost")
sigma=m.addVar(vtype=GRB.BINARY, name="Auxiliary_Variable")
psi=m.addVar(vtype=GRB.BINARY, name="Auxiliary_Variable")
beta=m.addVar(vtype=GRB.BINARY, name="Auxiliary_Variable")

#Defining the Dual variables, NOT SURE IF THIS IS CORRECT!!, If not then we'll change it later:

eta_gtw=m.addVar(type=GRB.CONTINUOUS, name="Dual variable for the quantity produced")
underline_gamma_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="Dual variable for the ramping limits")
Bar_gamma_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="Dual variable for the ramping limits")
lambda_w=m.addVar(vtype=GRB.CONTINUOUS, name="Dual variable for renewab")
phi_g=m.addVar(vtype=GRB.CONTINUOUS, name="Dual variable for the new capacity in generator")
delta_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="daul variable for the profit function ")
Thetagw=m.addVar(vtype=GRB.CONTINUOUS, name="daul variable for the auxilary variable")
muC_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="daul variable for revenue")
muR_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="daul variable for cost ")
l_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="daul variable for emissions")
psi_gtw=m.addVar(vtype=GRB.CONTINUOUS, name="daul variable for the quantity")

#Defining the parameters - Prices are in Euro 

piw=1 #Probability, for now it is 1 just to test one case   
tau=1 #Time, for now it is 1 just to test one case
vg=0 #This is confidence level for the generator, for now it is 0 just to test one case
beta=155 #intercept of demand function 
alpha=0.5 #slope of demand function
cg=5 #Marginal cost of production
ig=10000 #Cost of new generating technology 
#r_gdown = 0 #rump down  - not suere if we are going to use that 
#r_gup = 0 #rump up - not suere if we are going to use that
zeta = 0.2 #emission intensity 
rho_gtw = 0.6 #capacity facto 
qg0r = 100 #Initial capacity of the renewable generator
qg0c = 100 #Initial capacity of the conventioanl generator
kappa = 0.4 #global required renewable penetration 
PCO2=55 #Price of CO2 per ton
ecap=100 #Emission cap for generator
conjectural_variation = 0 # this can be either 0 or 1, for now it is 0 


#Defining the SOS constraints  https://www.gurobi.com/documentation/current/refman/py_model_addsos.html

#Defining the SOS constraints
#sos_vars = [eta_gtw, underline_gamma_gtw, Bar_gamma_gtw, lambda_w, phi_g, delta_gtw, Thetagw, muC_gtw, muR_gtw, l_gtw, psi_gtw]
#sos_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#m.addSOS(GRB.SOS_TYPE1, sos_vars, sos_weights)


#https://www.gurobi.com/documentation/current/refman/sos_constraints.html#subsubsection:SOSConstraints
