import tensorflow as tf
import scipy.special as ss

class spherical_harmonics:
    def __init__(self,n_max,m_max,M,R):
        self.n_max = n_max
        self.m_max = m_max
        self.R = R
        self.G = 6.6743e-11
        self.mu = self.G*M

    def J(self,n):
        if n==2: return -0.1082635854e-2

    def C(self,n,m):
        if n==2 and m==1: return -0.3504890360e-9
        if n==2 and m==2: return 0.1574536043e-5
        raise NotImplementedError
    
    def S(self,n,m):
        if n==2 and m==1: return 0.1635406077e-8
        if n==2 and m==2: return -0.9038680729e-6
        raise NotImplementedError
    
    def C_bar(self,n,m):
        return self.C(n,m)*-1/(self.mu*self.R**n)
    
    def S_bar(self,n,m):
        return self.S(n,m)*-1/(self.mu*self.R**n)
    
    def J_bar(self,n):
        return -self.J(n)/(self.mu*self.R**n)

    def P(self,n,m,x):
        term1 = (-1)**m*2**n
        term2 = (1-x**2)**(m/2)
        term3 = 0
        for k in range(m,n+1):
            term3 += ss.factorial(k)/ss.factorial(k-m)*x**(k-m)*ss.comb(n,k)*ss.comb((n+k-1)/2,n)
        return term1*term2*term3

    def potential(self,r):
        x,y,z = r[0],r[1],r[2]
        r_abs = tf.math.sqrt(x**2+y**2+z**2)
        theta = tf.math.asin(z/r_abs)
        phi = tf.math.acos(x/(r_abs*tf.math.cos(theta)))
        cs_term = 0
        for n in range(2,self.n_max+1):
            for m in range(1,n+1):
                cs_term += self.P(n,m,tf.math.sin(theta))*(self.C_bar(n,m)*tf.math.cos(m*phi)+self.S_bar(n,m)*tf.math.sin(m*phi))/((r_abs/self.R)**n)
        j_term = 0
        for n in range(2,self.n_max+1):
            j_term += self.J_bar(n)*self.P(n,0,tf.math.sin(theta))/((r_abs/self.R)**n)
        return -self.mu/r_abs*(1+cs_term+j_term)
    
    def acceleration(self,r):
        r = tf.convert_to_tensor(r)
        with tf.GradientTape() as tape:
            tape.watch(r)
            U = -1*self.potential(r)
        a = tape.gradient(U,r)
        return a.numpy()

if __name__=="__main__":
    sh = spherical_harmonics(n_max=2,m_max=2,M=5.972e24,R=6.371e6)
    r = [4585702.33424386, -677053.64507565, 4461705.57411378]
    # point mass would yield [ 6.8632174 -1.0133166  6.677639 ]
    print("acc from spherical harmonics\n",sh.acceleration(r))
    r = tf.convert_to_tensor(r)
    print("acc from point mass\n",(-sh.mu*r/(tf.norm(r)**3)).numpy())