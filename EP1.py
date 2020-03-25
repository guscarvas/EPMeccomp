import numpy as np
import matplotlib.pyplot as plt
# import math
from sympy import *
#
#   EULER
#  [ theta1[k+1], theta2[k+1], x1[k+1], x2[k+1] ] = [ theta1[k], theta2[k], x1[k], x2[k] ] + h*[ x1[k], x2[2], x_dot1[k], x_dot2[k] ]
#
#

class MatrizTudo(object): 
    def __init__(self, n,*args, **kwargs):
       self._matriz = np.zeros((6,n))
        # super(MatrizTudo, self).__init__(*args, **kwargs)
    
    def variaveis(self, k):
        return self._matriz[0:4,k]
    
    def derivadas(self, k):
        return self._matriz[2:6,k]
    
    def cond_iniciais(self,array):
        self._matriz[:,0] = array

def theta1_2dot(t,theta1,theta2,theta1_dot,theta2_dot,A0,A1,A2,A3,A4,A5):
    # A0 = (L1**2)*L2*R*(m2*math.cos(2*theta1-2*theta2)-2*m1-m2)
    # A1 = (L1**2)*L2*R*m2*math.sin(2*theta1-2*theta2)
    # A2 = 2*L1*(L2**2)*R*m2*math.sin(theta1-theta2)
    # A3 = -2*L2*mi*Iz*Vel
    # A4 = -2*L1*mi*Iz*Vel*math.cos(theta1-theta2)
    # A5 = -R*L1*(L2eixo*F2*math.sin(theta1-2*theta2)+2*math.sin(theta1)*(F1*L2+(L2eixo*F2)/2))
    return ((A1(theta1,theta2)*theta1_dot**2+A2(theta1,theta2)*theta2_dot**2+A3(theta1,theta2)*theta1_dot+A4(theta1,theta2)*theta2_dot + A5(theta1,theta2))/A0(theta1,theta2))

def theta2_2dot(t,theta1,theta2,theta1_dot,theta2_dot,theta1_2dotvar,B0,B1,B2,B3,B4):
    # B0 = (L2**2)*R*m2
    # B1 = -L1*L2*R*m2*math.cos(theta1-theta2)
    # B2 = L1*L2*R*m2*math.sin(theta1-theta2)
    # B3 = -mi*Iz*Vel
    # B4 = L2eixo*math.sin(theta2)*R*F2
    return ((B1(theta1,theta2)*theta1_2dotvar+B2(theta1,theta2)*theta1_dot**2+B3(theta1,theta2)*theta2_dot+B4(theta1,theta2))/B0(theta1,theta2))


# def euler(MatrizTd, h):

#     for i in range(n-1):
#         MatrizTd.variaveis(i+1) = MatrizTd.variaveis(i)+h*MatrizTd.derivadas(i)

def criaConstEqMov():
    L1=2
    L2=2.5
    L2eixo=1.8
    m1=450
    m2=650
    g=9.81
    F1=-0.5*m1*g
    F2=-0.5*m2*g
    miIz=2.7
    R=0.3
    Vel=80/3.6
    theta1 = symbols('theta1')
    theta2 = symbols('theta2')
    theta1_dot = symbols('theta1_dot')
    theta2_2dot = symbols('theta2_2dot')
    theta1_2dotvar = symbols('theta1_2dotvar')
    A0 = (L1**2)*L2*R*(m2*cos(2*theta1-2*theta2)-2*m1-m2)
    A1 = (L1**2)*L2*R*m2*sin(2*theta1-2*theta2)
    A2 = 2*L1*(L2**2)*R*m2*sin(theta1-theta2)
    A3 = -2*L2*miIz*Vel
    A4 = -2*L1*miIz*Vel*cos(theta1-theta2)
    A5 = -R*L1*(L2eixo*F2*sin(theta1-2*theta2)+2*sin(theta1)*(F1*L2+(L2eixo*F2)/2))
    B0 = (L2**2)*R*m2
    B1 = -L1*L2*R*m2*cos(theta1-theta2)
    B2 = L1*L2*R*m2*sin(theta1-theta2)
    B3 = -miIz*Vel
    B4 = L2eixo*sin(theta2)*R*F2
    array = np.array([[A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4],[0,0,0,0,0,0,0,0,0,0,0]])
    for i in range(11):
        temp = simplify(array[0,i])
        array[1,i] = lambdify((theta1,theta2),temp)
        print(i)
        print(array[1,i](0,0))
    return array[1]

def main():
    # n = (tf-ti)/h
    # MatrizTd = MatrizTudo(10)
    # MatrizTd.cond_iniciais([0,0.4,0,-0.1,theta1_2dot(0,0,0.4,0,-0.1)])
    # print(MatrizTd._matriz)
    # theta1 = np.zeros(n)
    # theta2 = np.zeros(n)
    # X1 = np.zeros(n)
    # X2 = np.zeros(n)
    [A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4] = criaConstEqMov()
    print(A0)
main()
