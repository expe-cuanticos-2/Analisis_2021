import numpy as np               
import pandas as pd              
import matplotlib.pyplot as plt  
import scipy.optimize as sts
import math as m
import scipy.integrate as integrate

#IMPORTAR DATOS___________________________________________________________________
df = pd.read_csv('C:\\Users\\nelib\\Desktop\\Exp II\\Gamma\\Ba133completo.csv')
names = ["0mm","1.92mm","2.92mm","4.21mm","9.91mm"]
espesores = [0,1.92,2.92,4.21,9.91]
a = 70
b = 170

esp = [ [  np.array(df.iloc[a:b,i]) ,names[i] ] for i in [0,1,2,3,4]   ]

#DEFINIR FUNCIONES
gauss = lambda x,mu,s,A: A*np.exp((-1/2)*((x-mu)/s)**2)
gauss_linear = lambda x,mu,s,A,a,b: A*np.exp((-1/2)*((x-mu)/s)**2) +  a*x+b
linear = lambda x,a,b: np.array(x)*a+b
area = lambda s,A: A*s*(2*m.pi)**(1/2)
area_err =  lambda s,A,s_err,A_err: ((A*(2*m.pi)**(1/2)*s_err)**2 + (s*(2*m.pi)**(1/2)*A_err)**2 )**(1/2)

#ANALISIS DE DATOS
x = range(a,b)
gauss_data = []
gauss_data_err = []
for i in esp[0:5]:
    error = np.sqrt(np.array(i[0]))
    plt.plot(x,i[0], label = i[1])    
    fit_iz = sts.curve_fit(gauss_linear,x, i[0], p0 = [110,10,600,-0.1,800], sigma=error)
    plt.plot(x,gauss_linear(x,fit_iz[0][0],fit_iz[0][1],fit_iz[0][2],fit_iz[0][3],fit_iz[0][4]))
    gauss_data.append([fit_iz[0][0],fit_iz[0][1],fit_iz[0][2]])
    gauss_data_err.append([fit_iz[1][0][0]**(1/2),fit_iz[1][1][1]**(1/2),fit_iz[1][2][2]**(1/2)])
plt.legend()
plt.xlabel("Canal")
plt.ylabel("Cuentas")
plt.show()


integrales_iz = [area(i[1],i[2]) for i in gauss_data]
integrales_err = [area_err(i[1],i[2],j[1],j[2]) for i,j in zip(gauss_data,gauss_data_err)]

"""""""""
#ANALISIS ESPECIFICO DE ANOMALIA
c = 50
d = 140
t = range(c,d)
y = np.array(df.iloc[c:d,4])
error = np.sqrt(y)
fit_ult =sts.curve_fit(gauss_linear, t,y,p0=[100,10,600,-0.1,800], sigma=error)
plt.plot(t, y)
plt.plot(t, gauss_linear(t,fit_ult[0][0],fit_ult[0][1],fit_ult[0][2],fit_ult[0][3],fit_ult[0][4]))
plt.plot(t, gauss(t,fit_ult[0][0],fit_ult[0][1],fit_ult[0][2]))
integrales_iz.append(area(fit_ult[0][2],fit_ult[0][3]))
integrales_err.append(area_err(fit_ult[0][2],fit_ult[0][3],fit_ult[1][2][2]**(1/2),fit_ult[1][3][3]**(1/2)))
plt.xlabel("Canal")
plt.ylabel("Cuentas")
plt.show()

"""""""""""

#AJUSTE LINEAL
g=5
logs = [ m.log(i/integrales_iz[0]) for i in integrales_iz[1:g]]
logs_err = np.array(integrales_err)/np.array(integrales_iz)
plt.errorbar(espesores[1:g], logs, yerr=logs_err[1:g], fmt="o")
fit_lin = sts.curve_fit(linear, espesores[1:g], logs, p0=[-1,0], sigma=logs_err[1:g])
plt.plot(espesores[1:g],linear(espesores[1:g],fit_lin[0][0],fit_lin[0][1]), label = f"mu={round(fit_lin[0][0],3)} +- {round(fit_lin[1][0][0]**(1/2),3)}")
plt.legend()
plt.ylabel("ln(I/I_0)")
plt.xlabel("Espesor [mm]")
plt.show()


print(f" el mu es = {round(fit_lin[0][0],3)} +- {round(fit_lin[1][0][0]**(1/2),3)}")
print(f"el porcentaje del error es = {(-fit_lin[1][0][0]**(1/2)/fit_lin[0][0])}")
print(logs_err)