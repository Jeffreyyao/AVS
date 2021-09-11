import numpy as np
from numpy import math

G = 6.67408e-11

# equations from wikipedia
def to_keplerian(rx, ry, rz, vx, vy, vz,M):
    r = np.array([rx,ry,rz])
    V = np.array([vx,vy,vz])
    r_abs = np.linalg.norm(r)
    V_abs = np.linalg.norm(V)
    mu = G*M
    # specific angular momentum vector
    h = np.cross(r,V)
    h_abs = np.linalg.norm(h)

    # specific energy
    E = V_abs**2/2-mu/r_abs

    # semi-major axis
    a = -mu/(2*E)

    # eccentricity vector
    e = np.cross(V,h)/mu-r/r_abs
    e_abs = np.linalg.norm(e)

    # inclination
    i = math.acos(h[2]/h_abs)

    # vector pointing to ascending node
    n = np.array([-h[1],h[0],0])
    n_abs = np.linalg.norm(n)

    # longitude of ascending node
    Omega = math.acos(n[0]/n_abs)
    if n[1]<0:
        Omega = 2*math.pi-Omega

    # argument of periapsis
    omega = math.acos(np.dot(n,e)/(n_abs*e_abs))
    if e[2]<0:
        omega = 2*math.pi-omega

    # true anomaly
    v = math.acos(np.dot(e,r)/(e_abs*r_abs))
    if np.dot(r,V)<0:
        v = 2*math.pi-v
    
    return (a,e_abs,i,Omega,omega,v)

#https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
def to_cartesian(a, e, i, Omega, omega, v, mass):
    mu = G*mass

    #https://space.stackexchange.com/questions/22144/difference-between-true-anomaly-and-mean-anomaly
    # eccentric anomaly
    E = math.acos((math.cos(v)+e)/(1+e*math.cos(v)))

    # distance to central body
    r_abs = a*(1-e*math.cos(E))

    # velocity components
    ox = r_abs*np.cos(v)
    oy = r_abs*np.sin(v)
    # acceleration components
    oxd = math.sqrt(mu*a)*-math.sin(E)/r_abs
    oyd = math.sqrt(mu*a)*math.sqrt(1-e**2)*math.cos(E)/r_abs

    # transformation to heliocentric inertial frame
    rx = ox*(math.cos(omega)*math.cos(Omega)-math.sin(omega)*math.cos(i)*math.sin(Omega))-oy*(math.sin(omega)*math.cos(Omega)+math.cos(omega)*math.cos(i)*math.sin(Omega))

    ry = ox*(math.cos(omega)*math.sin(Omega)+math.sin(omega)*math.cos(i)*math.cos(Omega))+oy*(math.cos(omega)*math.cos(i)*math.cos(Omega)-math.sin(omega)*math.sin(Omega))

    rz = ox*math.sin(omega)*math.sin(i)+oy*math.cos(omega)*math.sin(i)

    vx = oxd*(math.cos(omega)*math.cos(Omega)-math.sin(omega)*math.cos(i)*math.sin(Omega))-oyd*(math.sin(omega)*math.cos(Omega)+math.cos(omega)*math.cos(i)*math.sin(Omega))

    vy = oxd*(math.cos(omega)*math.sin(Omega)+math.sin(omega)*math.cos(i)*math.cos(Omega))+oyd*(math.cos(omega)*math.cos(i)*math.cos(Omega)-math.sin(omega)*math.sin(Omega))

    vz = oxd*math.sin(omega)*math.sin(i)+oyd*math.cos(omega)*math.sin(i)

    return (rx,ry,rz,vx,vy,vz)

if __name__=="__main__":
    (a,e,i,Omega,omege,v) = to_keplerian(1e9,2e9,3e9,3000,2000,1000,1.99e30)
    print(to_cartesian(a,e,i,Omega,omege,v,1.99e30))