
# Importation des modules
import numpy as np
import matplotlib.pyplot as plt

#%%MEC6616-Aérodynamique numérique


#Fait par:
    #Nicolas Tran
    #Mathis Turquet


#%% Assisgnation des constantes donnés dans l'énoncé

class parametres1():
    L = .5     # Longueur
    x_0 = 0
    x_f = L
    T_A = 100  # Temperature à x=A
    T_B = 500  # Temperature à x=B
    k = 1000   # Conductivité thermique  W/m.K
    A = 10e-3  # Section  m^2
    N =  5    #Nombre de noeuds
    
class parametres2():
    L = .02    # Longueur
    x_0 = 0
    x_f = L
    T_A = 100  # Temperature à x=A
    T_B = 200  # Temperature à x=B
    k = .5     # Conductivité thermique  W/m.K
    q = 1000000 #heat source (W/m^3)
    A = 1 # Section  m^2
    N =  5    #Nombre de noeuds

class parametres3():
    L = 1     # Longueur
    x_0 = 0
    x_f = L
    T_inf = 20  # Temperature à x=A
    T_B = 100  # Temperature à x=B
    #k = 1000   # Conductivité thermique  W/m.K
    #h = #convective heat transfert coeff (W`m[2.K])
    #P = #Perimeter
    #A = 10e-3  # Section  m^2
    N =  5    #Nombre de noeuds    
    n_2=25    #Coeff thermique
    
    
prm1 = parametres1()
prm2 = parametres2()
prm3 = parametres3()

#%% Fonctions
def D1HT(x,prm,n):
    """Fonction 
    Entrées:
        - prm : Objet class parametres()
               - L : Longueur
               - T_A : Temperature à x=A
               - T_B : Temperature à x=B
               - k : Conductivité thermique  W/m.K
               - A : Section  m^2
        - Nombre de noeuds
    Sortie:
        - Vecteur (array) composé de la température en fonction de la position
            en régime stationnaire
    """
    
    # Constantes
    L = prm.L
    T_A = prm.T_A
    T_B = prm.T_B
    k = prm.k
    A = prm.A
    dx = L/(n)
    
    #Equations aux noeuds internes
    a_w_int=k/dx*A
    a_e_int=k/dx*A
    S_u_int=0
    a_p_int=a_w_int+a_e_int
    
    #Equations au noeud 1 (A)
    a_w_A=0
    a_e_A=k/dx*A
    S_p_A=-2*k*A/dx
    S_u_A=2*k*A/dx*T_A
    a_p_A=a_w_A+a_e_A-S_p_A
    
    #Equations au noeud 5 (B)
    a_w_B=k/dx*A
    a_e_B=0
    S_p_B=-2*k*A/dx
    S_u_B=2*k*A/dx*T_B
    a_p_B=a_w_B+a_e_B-S_p_B
    
    # Créer la matrice de températures
    T = np.zeros(n)

    # Conditions limites et matrice b
    b = np.zeros(n)
    b[0:n] = S_u_int
    b[0] = S_u_A
    b[-1] = S_u_B

    # Matrice A
    A = np.zeros((n,n))
    A[0,0] = a_p_A
    A[0,1] = -a_e_A
    A[-1,-1] = a_p_B
    A[-1,-2] = -a_w_B
    for i in range(1, n-1):
        A[i,i-1] = -a_w_int
        A[i,i] = a_p_int
        A[i,i+1] = -a_e_int

    # Résolution du système matriciel AT=b
    T = np.linalg.solve(A,b)
    return T


def analytique1(x,prm,n):
    """Fonction 
    Entrées:
        - prm : Objet class parametres()
               - L : Longueur
               - T_A : Temperature à x=A
               - T_B : Temperature à x=B
               - k : Conductivité thermique  W/m.K
               - A : Section  m^2
        - N = Nombre de noeuds
    Sortie:
        - Vecteur (array) composé de la température en fonction de la position
            en régime stationnaire de la solution analytique
    """
    
    T = np.zeros(n)
    for i in range(n):
        T[i]=800*x[i]+100
    
    return T

def erreur_L2(sol_num, sol_anal):
    return np.sqrt(np.sum((sol_num - sol_anal)**2) / len(sol_num))

def erreur_Linf(sol_num, sol_anal):
    return np.max(np.abs(sol_num - sol_anal))

def OrdreConvergeance(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    slope = dy / dx
    return slope


def QHT(x,prm,n):
    
    #Paramètres
    L = prm.L
    T_A = prm.T_A
    T_B = prm.T_B
    k = prm.k
    A = prm.A
    q = prm.q
    dx = L/(n)
    
    #Equations aux noeuds internes
    a_w_int=k/dx*A
    a_e_int=k/dx*A
    S_p_int=0
    S_u_int=q*A*dx
    a_p_int=a_w_int+a_e_int-S_p_int
    
    #Equations au noeud 1
    a_w_A=0
    a_e_A=k/dx*A
    S_p_A=-2*k*A/dx
    S_u_A=q*A*dx+ 2*k*A/dx*T_A
    a_p_A=a_w_A+a_e_A-S_p_A
    
    #Equations au noeud 5
    a_w_B=k/dx*A
    a_e_B=0
    S_p_B=-2*k*A/dx
    S_u_B=q*A*dx+2*k*A/dx*T_B
    a_p_B=a_w_B+a_e_B-S_p_B
    
    # Créer la matrice de températures
    T = np.zeros(n)

    # Conditions limites et matrice b
    b = np.zeros(n)
    b[0:n] = S_u_int
    b[0] = S_u_A
    b[-1] = S_u_B

    # Matrice A
    A = np.zeros((n,n))
    A[0,0] = a_p_A
    A[0,1] = -a_e_A
    A[-1,-1] = a_p_B
    A[-1,-2] = -a_w_B
    for i in range(1, n-1):
        A[i,i-1] = -a_w_int
        A[i,i] = a_p_int
        A[i,i+1] = -a_e_int

    # Résolution du système matriciel AT=b
    T = np.linalg.solve(A,b)
    return T


def analytique2(x,prm,n):
    """Fonction 
    Entrées:
        - prm : Objet class parametres()
               - L : Longueur
               - T_A : Temperature à x=A
               - T_B : Temperature à x=B
               - k : Conductivité thermique  W/m.K
               - A : Section  m^2
        - N = Nombre de noeuds
    Sortie:
        - Vecteur (array) composé de la température en fonction de la position
            en régime stationnaire de la solution analytique
    """
    #Paramètres
    L = prm.L
    T_A = prm.T_A
    T_B = prm.T_B
    k = prm.k
    q = prm.q

    # Résolution analytique
    T = np.zeros(n)
    for i in range(n):
        T[i]=((T_B-T_A)/L+q/(2*k)*(L-x[i]))*x[i]+T_A
    
    return T

def ConvHT(x,prm,n):
    
    #Paramètres
    L = prm.L
    T_inf = prm.T_inf
    T_B = prm.T_B
    n_2 = prm.n_2
    dx = L/(n)
    
    #Equations aux noeuds internes
    a_w_int=1/dx
    a_e_int=1/dx
    S_p_int=-n_2*dx
    S_u_int=n_2*dx*T_inf
    a_p_int=a_w_int+a_e_int-S_p_int
    
    #Equations au noeud 1
    a_w_A=0
    a_e_A=1/dx
    S_p_A=-n_2*dx-2/dx
    S_u_A=n_2*dx*T_inf+2/dx*T_B
    a_p_A=a_w_A+a_e_A-S_p_A
    
    #Equations au noeuds 5
    a_w_B=1/dx
    a_e_B=0
    S_p_B=-n_2*dx
    S_u_B=n_2*dx*T_inf
    a_p_B=a_w_B+a_e_B-S_p_B
    
    # Créer la matrice de températures
    T = np.zeros(n)

    # Conditions limites et matrice b
    b = np.zeros(n)
    b[0:n] = S_u_int
    b[0] = S_u_A
    b[-1] = S_u_B

    # Matrice A
    A = np.zeros((n,n))
    A[0,0] = a_p_A
    A[0,1] = -a_e_A
    A[-1,-1] = a_p_B
    A[-1,-2] = -a_w_B
    for i in range(1, n-1):
        A[i,i-1] = -a_w_int
        A[i,i] = a_p_int
        A[i,i+1] = -a_e_int

    # Résolution du système matriciel AT=b
    T = np.linalg.solve(A,b)
    return T
    
def analytique3(x,prm,n):
    """Fonction 
    Entrées:
        - prm : Objet class parametres()
               - L : Longueur
               - T_A : Temperature à x=A
               - T_B : Temperature à x=B
               - k : Conductivité thermique  W/m.K
               - A : Section  m^2
        - N = Nombre de noeuds
    Sortie:
        - Vecteur (array) composé de la température en fonction de la position
            en régime stationnaire de la solution analytique
    """
    #Paramètres
    L = prm.L
    T_inf = prm.T_inf
    T_B = prm.T_B
    n_2 = prm.n_2

    # Résolution analytique
    T = np.zeros(n)
    for i in range(n):
        T[i]= np.cosh(np.sqrt(n_2)*(L-x[i]))/np.cosh(np.sqrt(n_2)*(L))*(T_B-T_inf)+T_inf
        
    return T


#%% Problème 4.1

# Conditions initiales
dx1=prm1.L/prm1.N

# Graphiques
x = np.linspace(prm1.x_0 + dx1 / 2, prm1.x_f - dx1 / 2, prm1.N)
T = D1HT(x, prm1, prm1.N)
x_anal = np.linspace(prm1.x_0 + dx1 / 2, prm1.x_f - dx1 / 2, prm1.N)
T_anal = analytique1(x_anal, prm1, prm1.N)
plt.figure(1)
plt.plot(x, T, '.', label='Solution numérique')
plt.plot(x_anal, T_anal, label='Solution analytique' )
plt.title('Solutions du problème 4.1 à n=5')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.xlim(0, prm1.L)
plt.legend()
# Erreur
# 

n_values = np.arange(5, 1001, 5)  
h_values = prm1.L / n_values  
Erreur_L2 = []
Erreur_Linf = []

for n in n_values:
    dx1 = prm1.L / n
    x_e = np.linspace(prm1.L / n / 2, prm1.L - prm1.L / n / 2, n)
    
    # Solution numérique et analytique
    sol_num = D1HT(x_e, prm1, n)
    sol_anal = analytique1(x_e, prm1, n)
    
    # Calcul des erreurs
    erreur_L2_value = erreur_L2(sol_num, sol_anal)
    erreur_Linf_value = erreur_Linf(sol_num, sol_anal)
    
    Erreur_L2.append(erreur_L2_value)
    Erreur_Linf.append(erreur_Linf_value)
    
plt.figure(2)
plt.loglog(h_values, Erreur_L2, label="Erreur L2")
plt.loglog(h_values, Erreur_Linf, label="Erreur Linfini")
plt.xlabel("h")
plt.ylabel("Erreur(h)")
plt.legend()
plt.grid(True)
plt.title("Convergence de l'erreur du problème 4.1")
plt.show()

A=str(max(abs(OrdreConvergeance(np.log(h_values), np.log(Erreur_Linf)))))
print("La convergeance de l'erreur pour le problème 4.1 est de :"+ A)
#%% Problème 4.2
dx2=prm2.L/prm2.N


x2 = np.linspace(prm2.x_0 + dx2 / 2, prm2.x_f - dx2 / 2, prm2.N)
T2 = QHT(x2, prm2, prm2.N)
x_anal2 = np.linspace(prm2.x_0 + dx2 / 2, prm2.x_f - dx2 / 2, prm2.N)
T_anal2 = analytique2(x_anal2, prm2, prm2.N)
plt.figure(3)
plt.plot(x2, T2, '.', label='Solution numérique')
plt.plot(x_anal2, T_anal2, label='Solution analytique' )
plt.title('Solutions du problème 4.2 à n=5')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.xlim(0, prm2.L)
plt.legend()

n_values = np.arange(5, 1001, 5)  
h_values = prm2.L / n_values  
Erreur_L2 = []
Erreur_Linf = []

for n in n_values:
    dx2 = prm2.L / n
    x_e = np.linspace(prm2.L/n/2, prm2.L - prm2.L/n/2, n)
    
    # Solution numérique et analytique
    sol_num = QHT(x_e, prm2, n)
    sol_anal = analytique2(x_e, prm2, n)
    
    # Calcul des erreurs
    erreur_L2_value = erreur_L2(sol_num, sol_anal)
    erreur_Linf_value = erreur_Linf(sol_num, sol_anal)
    
    Erreur_L2.append(erreur_L2_value)
    Erreur_Linf.append(erreur_Linf_value)
    
plt.figure(4)
plt.loglog(h_values, Erreur_L2, label="Erreur L2")
plt.loglog(h_values, Erreur_Linf, label="Erreur Linfini")
plt.xlabel("h")
plt.ylabel("Erreur(h)")
plt.legend()
plt.grid(True)
plt.title("Convergence de l'erreur du problème 4.2")
plt.show()

B=str(max(abs(OrdreConvergeance(np.log(h_values), np.log(Erreur_Linf)))))
print("La convergeance de l'erreur pour le problème 4.2 est de :"+ B)
 #%% Problème 4.3
dx3=prm3.L/prm3.N


x3 = np.linspace(prm3.x_0 + dx3 / 2, prm3.x_f - dx3 / 2, prm3.N)
T3 = ConvHT(x3, prm3, prm3.N)
x_anal3 = np.linspace(prm3.x_0 + dx3 / 2, prm3.x_f - dx3 / 2, prm2.N)
T_anal3 = analytique3(x_anal3, prm3, prm3.N)
plt.figure(5)
plt.plot(x3, T3, '.', label='Solution numérique')
plt.plot(x_anal3, T_anal3, label='Solution analytique' )
plt.title('Solutions du problème 4.3 à n=5')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.xlim(0, prm3.L)
plt.legend()
n_values = np.arange(5, 1001, 5)  
h_values = prm2.L / n_values  
Erreur_L2 = []
Erreur_Linf = []

for n in n_values:
    dx2 = prm3.L / n
    x_e = np.linspace(prm3.L/n/2, prm3.L - prm3.L/n/2, n)
    
    # Solution numérique et analytique
    sol_num = ConvHT(x_e, prm3, n)
    sol_anal = analytique3(x_e, prm3, n)
    
    # Calcul des erreurs
    erreur_L2_value = erreur_L2(sol_num, sol_anal)
    erreur_Linf_value = erreur_Linf(sol_num, sol_anal)
    
    Erreur_L2.append(erreur_L2_value)
    Erreur_Linf.append(erreur_Linf_value)
    
plt.figure(6)
plt.loglog(h_values, Erreur_L2, label="Erreur L2")
plt.loglog(h_values, Erreur_Linf, label="Erreur Linfini")
plt.xlabel("h")
plt.ylabel("Erreur(h)")
plt.legend()
plt.grid(True)
plt.title("Convergence de l'erreur du problème 4.3")
plt.show()
C=str(max(abs(OrdreConvergeance(np.log(h_values), np.log(Erreur_Linf)))))
print("La convergeance de l'erreur pour le problème 4.3 est de :"+ C)


