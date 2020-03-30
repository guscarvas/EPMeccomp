import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
from sympy import *
import matplotlib.animation as animation
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


def eulerMethod(MatrizTd, h, n, A, B):

    for k in range(n-1):
        theta1_2dot = calc_theta1_2dot(MatrizTd.theta1(k),MatrizTd.theta2(k),MatrizTd.theta1_dot(k),MatrizTd.theta2_dot(k),A)
        MatrizTd.set_theta1_2dot(theta1_2dot,k)
        theta2_2dot = calc_theta2_2dot(MatrizTd.theta1(k),MatrizTd.theta2(k),MatrizTd.theta1_dot(k),MatrizTd.theta2_dot(k),theta1_2dot,B)
        MatrizTd.set_theta2_2dot(theta2_2dot,k)

        proxIterVar = np.add(MatrizTd.variaveis(k),np.multiply(MatrizTd.derivadas(k),h))
        # proxIterVar = MatrizTd.variaveis(k)+h*MatrizTd.derivadas(k)
        MatrizTd.set_variaveis(proxIterVar,k+1)

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
        
        Mtd.set_theta1_2dot(theta1_2dot,k)
        Mtd.set_theta2_2dot(theta2_2dot,k)

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
        l2 = lx(theta1+h*k1,theta2+h*m1,theta1_dot+h*l1,theta2_dot+h*n1)
        ############## M2
        m2 = mx(theta2_dot+h*n1)
        ############## N2
        n2 = nx(theta1+h*k1,theta2+h*m1,theta1_dot+h*l1,theta2_dot+h*n1,l2)
        
        ############CALCULO PROXIMA ITERAÇÃO
        
        somas = np.array([(k1+k2)/2,(m1+m2)/2,(l1+l2)/2,(n1+n2)/2])

        proxIterVar = np.add(Mtd.variaveis(k),np.multiply(somas,h))
        Mtd.set_variaveis(proxIterVar,k+1)

def rungeKutta4(Mtd,h,n,A,B):
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
        
        Mtd.set_theta1_2dot(theta1_2dot,k)
        Mtd.set_theta2_2dot(theta2_2dot,k)

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
        k2 = kx(theta1_dot+(h/2)*l1)
        ############## L2
        l2 = lx(theta1+(h/2)*k1,theta2+(h/2)*m1,theta1_dot+(h/2)*l1,theta2_dot+(h/2)*n1)
        ############## M2
        m2 = mx(theta2_dot+(h/2)*n1)
        ############## N2
        n2 = nx(theta1+(h/2)*k1,theta2+(h/2)*m1,theta1_dot+(h/2)*l1,theta2_dot+(h/2)*n1,l2)
        
        ####### Calculo de #3

        ############## K3
        k3 = kx(theta1_dot+(h/2)*l2)
        ############## L3
        l3 = lx(theta1+(h/2)*k2,theta2+(h/2)*m2,theta1_dot+(h/2)*l2,theta2_dot+(h/2)*n2)    
        ############## M3
        m3 = mx(theta2_dot+(h/2)*n2)
        ############## N3
        n3 = nx(theta1+(h/2)*k2,theta2+(h/2)*m2,theta1_dot+(h/2)*l2,theta2_dot+(h/2)*n2,l3)
        ####### Calculo de #4
        
        ############## K4
        k4 = kx(theta1_dot+h*l3)
        ############## L4
        l4 = lx(theta1+h*k3,theta2+h*m3,theta1_dot+h*l3,theta2_dot+h*n3)
        ############## M4
        m4 = mx(theta2_dot+h*n3)
        ############## N4
        n4 = nx(theta1+h*k3,theta2+h*m3,theta1_dot+h*l3,theta2_dot+h*n3,l4)
        
        ############CALCULO PROXIMA ITERAÇÃO
        
        somas = np.array([(k1+2*k2+2*k3+k4)/6,(m1+2*m2+2*m3+m4)/6,(l1+2*l2+2*l3+l4)/6,(n1+2*n2+2*n3+n4)/6])

        proxIterVar = np.add(Mtd.variaveis(k),np.multiply(somas,h))
        Mtd.set_variaveis(proxIterVar,k+1)


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
    return array[1,:6],array[1, 6:]

def main():
    L1=2
    L2=2.5
    L2eixo=1.8

    theta1_i = 0
    theta2_i = 0
    theta1dot_i = 0.4
    theta2dot_i = -0.1

    print("Selecione o método que deseja realizar: \n0 - Método de Euler \n1 - RK2 \n2 - RK4")
    metodo = int(input("Sua escolha: "))
    h = float(input("Defina o tamanho do passo que será utilizado: "))
    tf = 60
    ti = 0
    n = int((tf-ti)/h)
    A, B = criaConstEqMov()
    MatrizTd = MatrizTudo(n)
    temp = calc_theta1_2dot(theta1_i,theta2_i,theta1dot_i,theta2dot_i,A)
    MatrizTd.cond_iniciais([theta1_i,theta2_i,theta1dot_i,theta2dot_i,temp,calc_theta2_2dot(theta1_i,theta2_i,theta1dot_i,theta2dot_i,temp,B)])

    if metodo == 0:
        print("Você selecionou método de Euler com passo",h)
        eulerMethod(MatrizTd,h,n, A, B)
        textoGrafico = 'Euler'

    elif metodo == 1:
        rungeKutta2(MatrizTd,h,n,A,B)
        textoGrafico = 'Euler Modificado/RK2'

    elif metodo == 2:
        rungeKutta2(MatrizTd,h,n,A,B)
        textoGrafico = 'RK4'

    ############ Inicio plotagem e Animação
    t = np.linspace(ti,tf,n) #vetor tempo
    

    plt.figure(1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.45)
    subplot(3,2,1)
    plt.title(r"${\Theta}_1$ para "+textoGrafico+ " com passo " + str(h))
    plt.xlabel('t (s)')
    # plt.ylabel('theta1 (rad)')
    plt.ylabel(r"${\Theta}_1$[rad]")
    plt.plot(t,MatrizTd._matriz[0])
    
    subplot(3,2,2)
    plt.title(r"${\Theta}_2$ para "+textoGrafico+ " com passo " + str(h))
    plt.xlabel('t (s)')
    plt.ylabel(r"${\Theta}_2$[rad]")
    plt.plot(t,MatrizTd._matriz[1])
    
    subplot(3,2,3)
    plt.title(r"$\dot{\Theta}_1$ para "+textoGrafico+ " com passo " + str(h))
    plt.xlabel('tempo(s)')
    plt.ylabel(r"$\dot{\Theta}_1$[rad/s]")
    plt.plot(t,MatrizTd._matriz[2])
    
    subplot(3,2,4)
    plt.title(r"$\dot{\Theta}_2$ para "+textoGrafico+ " com passo " + str(h))
    plt.xlabel('tempo(s)')
    plt.ylabel(r"$\dot{\Theta}_2$[rad/s]")
    plt.plot(t,MatrizTd._matriz[3])
    
    subplot(3,2,5)
    plt.title(r"$\ddot{\Theta}_1$ para "+textoGrafico+ " com passo " + str(h))
    plt.xlabel('tempo(s)')
    plt.ylabel(r"$\ddot{\Theta}_1[rad/s^2]$")
    plt.plot(t,MatrizTd._matriz[4])
    
    subplot(3,2,6)
    plt.title(r"$\ddot{\Theta}_2$ para "+textoGrafico+ " com passo " + str(h))
    plt.xlabel('tempo(s)')
    plt.ylabel(r"$\ddot{\Theta}_2$[rad/s^2]$")
    plt.plot(t,MatrizTd._matriz[5])

    x1 = L1*np.sin(MatrizTd._matriz[0])
    y1 = -L1*np.cos(MatrizTd._matriz[0])

    x1rodaEsquerda = 0.75*np.cos(MatrizTd._matriz[0]) + x1
    y1rodaEsquerda = 0.75*np.sin(MatrizTd._matriz[0]) + y1
    x1rodaDireita = -0.75*np.cos(MatrizTd._matriz[0]) + x1
    y1rodaDireita = -0.75*np.sin(MatrizTd._matriz[0]) + y1
    
    x2 = L2eixo*np.sin(MatrizTd._matriz[1]) + x1
    y2 = -L2eixo*np.cos(MatrizTd._matriz[1]) + y1

    x2rodaEsquerda = 0.75*np.cos(MatrizTd._matriz[1]) + x2
    y2rodaEsquerda = 0.75*np.sin(MatrizTd._matriz[1]) + y2
    x2rodaDireita = -0.75*np.cos(MatrizTd._matriz[1]) + x2
    y2rodaDireita = -0.75*np.sin(MatrizTd._matriz[1]) + y2

    x2eixo = L2*np.sin(MatrizTd._matriz[1]) + x1
    y2eixo = -L2*np.cos(MatrizTd._matriz[1]) + y1

    fig = plt.figure(2)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 5), ylim=(-6, 0))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text


    def animate(i):
        thisx = [0, x1[i],x1rodaDireita[i],x1rodaEsquerda[i],x1[i], 
                x2[i], x2rodaDireita[i], x2rodaEsquerda[i], x2[i], x2eixo[i]]
        thisy = [0, y1[i],y1rodaDireita[i],y1rodaEsquerda[i],y1[i], y2[i],y2rodaDireita[i], y2rodaEsquerda[i], y2[i], y2eixo[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (t[i]))
        return line, time_text
    

    # ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t)),
    #                           interval=0.005, blit=True, init_func=init)

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t)),
                              interval=5, blit=True, init_func=init)

    
    # ani.save('Ep1.mp4', fps=15)

    plt.show()

    
    
main()
