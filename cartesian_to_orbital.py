import numpy as np

# equations from wikipedia
def convert(r,V,M,m):
    r_abs = np.linalg.norm(r)
    V_abs = np.linalg.norm(V)
    G = 6.67408e-11
    mu = G*(M+m)
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
    i = np.math.acos(h[2]/h_abs)

    # vector pointing to ascending node
    n = np.array([-h[1],h[0],0])
    n_abs = np.linalg.norm(n)

    # longitude of ascending node
    Omega = np.math.acos(n[0]/n_abs)
    if n[1]<0:
        Omega = 2*np.math.pi-Omega

    # argument of periapsis
    omega = np.math.acos(np.dot(n,e)/(n_abs*e_abs))
    if e[2]<0:
        omega = 2*np.math.pi-omega

    # true anomaly
    v = np.math.acos(np.dot(e,r)/(e_abs*r_abs))
    if np.dot(r,V)<0:
        v = 2*np.math.pi-v
    
    return (a,e_abs,i,Omega,omega,v)

if __name__=="__main__":
    print(convert(np.array([1e10,2e10,3e10]),np.array([30,20,10]),2e30,1e30))