import numpy as np
import matplotlib.pyplot as plt
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

    def set_variaveis(self,array, k):
        self._matriz[0:4,k] = array
    
    def derivadas(self, k):
        return self._matriz[2:6,k]
    
    def cond_iniciais(self,array):
        self._matriz[:,0] = array

    def theta1(self,k):
        return self._matriz[0,k]

    def theta2(self,k):
        return self._matriz[1,k]

    def theta1_dot(self,k):
        return self._matriz[2,k]

    def theta2_dot(self,k):
        return self._matriz[3,k]

    def theta1_2dot(self,k):
        return self._matriz[4,k]

    def set_theta1_2dot(self,value,k):
        self._matriz[4,k] = value

    def theta2_2dot(self,k):
        return self._matriz[5,k]

    def set_theta2_2dot(self,value,k):
        self._matriz[5,k] = value


def calc_theta1_2dot(theta1,theta2,theta1_dot,theta2_dot,A):
    # A0 = (L1**2)*L2*R*(m2*math.cos(2*theta1-2*theta2)-2*m1-m2)
    # A1 = (L1**2)*L2*R*m2*math.sin(2*theta1-2*theta2)
    # A2 = 2*L1*(L2**2)*R*m2*math.sin(theta1-theta2)
    # A3 = -2*L2*mi*Iz*Vel
    # A4 = -2*L1*mi*Iz*Vel*math.cos(theta1-theta2)
    # A5 = -R*L1*(L2eixo*F2*math.sin(theta1-2*theta2)+2*math.sin(theta1)*(F1*L2+(L2eixo*F2)/2))
    return ((A[1](theta1,theta2)*theta1_dot**2+A[2](theta1,theta2)*theta2_dot**2+A[3](theta1,theta2)*theta1_dot+A[4](theta1,theta2)*theta2_dot + A[5](theta1,theta2))/A[0](theta1,theta2))

def calc_theta2_2dot(theta1,theta2,theta1_dot,theta2_dot,theta1_2dotvar,B):
    # B0 = (L2**2)*R*m2
    # B1 = -L1*L2*R*m2*math.cos(theta1-theta2)
    # B2 = L1*L2*R*m2*math.sin(theta1-theta2)
    # B3 = -mi*Iz*Vel
    # B4 = L2eixo*math.sin(theta2)*R*F2
    return ((B[1](theta1,theta2)*theta1_2dotvar+B[2](theta1,theta2)*theta1_dot**2+B[3](theta1,theta2)*theta2_dot+B[4](theta1,theta2))/B[0](theta1,theta2))


# def euler(MatrizTd, h):

#     for i in range(n-1):
#         MatrizTd.variaveis(i+1) = MatrizTd.variaveis(i)+h*MatrizTd.derivadas(i)


def rungeKutta2(Mtd,h,n,A,B):
    def kx(theta1_dot):
        return theta1_dot

    def lx(theta1,theta2,theta1_dot,theta2_dot):
        return calc_theta1_2dot(theta1,theta2,theta1_dot,theta2_dot,A)
    
    def mx(theta2_dot):
        return theta2_dot

    def nx(theta1,theta2,theta1_dot,theta2_dot,theta1_2dotvar):
        return calc_theta2_2dot(theta1,theta2,theta1_dot,theta2_dot,theta1_2dotvar,B)
    
    for k in range(n-1):
        [theta1,theta2,theta1_dot,theta2_dot] = Mtd.variaveis(k)

        theta1_2dot = calc_theta1_2dot(theta1,theta2,theta1_dot,theta2_dot,A)
        theta2_2dot = calc_theta2_2dot(theta1,theta2,theta1_dot,theta2_dot,theta1_2dot,B)
        
        Mtd.set_theta1_2dot(theta1_2dot)
        Mtd.set_theta2_2dot(theta2_2dot)

        ####### Calculo de #1
        ############## K1
        k1 = kx(theta1_dot)
        ############## L1
        l1 = theta1_2dot
        ############## M1
        m1 = mx(theta2_dot)
        ############## N1
        n1 = theta2_2dot
        ####### Calculo de #2
        ############## K2
        k2 = kx(theta1_dot+h*l1)
        ############## L2
        l2 = lx(theta1+h*k1,theta2+h*m1,theta1_dot+h*l1,theta2_dot*n1)
        ############## M2
        m2 = mx(theta2_dot+n1)
        ############## N2
        n2 = nx(theta1+h*k1,theta2+h*m1,theta1_dot+h*l1,theta2_dot*n1,l2)
        
        ############CALCULO PROXIMA ITERAÇÃO
        
        somas = np.array([(k1+k2)/2,(m1+m2)/2,(l1+l2)/2],(n1+n2)/2)

        proxIterVar = np.add(Mtd.variaveis(k),np.multiply(somas,h))
        Mtd.set_variaveis(proxIterVar,k+1)
        ####### Calculo de #3

        ############## K3
        
        ############## L3

        ############## M3

        ############## N3

        ####### Calculo de #4

        ############## K4

        ############## L4

        ############## M4

        ############## N4




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
    return array[1,:6],array[1, 6:]

def main():
    tf = 60
    ti = 0
    h = 0.01
    n = (tf-ti)/h
    A, B = criaConstEqMov()
    MatrizTd = MatrizTudo(n)
    temp = calc_theta1_2dot(0,0.4,0,-0.1,A)
    MatrizTd.cond_iniciais([0,0.4,0,-0.1,temp,calc_theta2_2dot(0,0.4,0,-0.1,temp,B)])
    
    
    # print(MatrizTd._matriz)
    # theta1 = np.zeros(n)
    # theta2 = np.zeros(n)
    # X1 = np.zeros(n)
    # X2 = np.zeros(n)

main()
