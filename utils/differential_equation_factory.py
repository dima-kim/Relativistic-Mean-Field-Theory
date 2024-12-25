import numpy as np

class coulomb_differential_equation:
    '''
    The U and V derivative functions for the coulomb coupled first order
    differential equations. Defined by the following
    

                                    du(r)/dr = U(r,u(r),v(r))
                                    dv(r)/dr = V(r,u(r),v(r))
    
                                        u(r) = g(r) / r
                                        v(r) = f(r) / r
    
    We are using the same conventions as in the book: "Relativistic Quantum Mechanics" by 
    P. Strange.

    Parameters
    ----------
    m : double
      The mass of my particle in [MeV].
    
    xi : double
      The coulomb potential strength
    
    kappa: int
      Kappa is related to the eigenvalue of the K operator.
      It is realted to total angular momentum.

    '''
    
    def __init__(self,m,xi,kappa):
        self.m = m
        self.hbarc = 197.33
        # self.hbarc = 197.326
        self.xi = xi
        self.kappa = kappa


    def U_coulomb(self):
        '''
        The derivative function of u(r), determined from the coulomb differential equation.

        Returns
        -------
        U : lambda function
          The first derivative of u(r). Inputs are kappa, r, u, v, and E

        '''

        U = lambda r, u, v, E: (- (self.kappa / r) * self.hbarc * u + (E + (self.xi / r) * self.hbarc + self.m) * v)

        return U

    def V_coulomb(self):
        '''
        The derivative function of v(r), determined from the coulomb differential equation.

        Returns
        -------
        V : lambda function
          The first derivative of v(r). Inputs are kappa, r, u, v, and E
          
        '''
        V = lambda r, u, v, E: (self.kappa / r) * self.hbarc * v - (E + (self.xi/ r) * self.hbarc - self.m) * u

        return V
    
    def initial_conditions(self, limit = ['zero','inf']):
        '''
        Returns the initial conditions for the wavefunctions.

        Returns
        -------
        u_initial : lambda func
          The u-wavefunction initial conditions.

        v_initial : lambda func
          The v-wavefunction initial conditions.
        '''
        
        u_initial = lambda r, E: 1E-6
        v_initial = lambda r, E: -1E-6
        
        return u_initial, v_initial
    
class RMF_differential_equation:
    '''
    The U and V derivative functions for the RMF coupled first order
    differential equations. Defined by the following
    

                                    du(r)/dr = U(r,u(r),v(r))
                                    dv(r)/dr = V(r,u(r),v(r))
    
                                        u(r) = g(r) / r
                                        v(r) = f(r) / r
    
    We are using the same conventions as in the book: "Relativistic Quantum Mechanics" by 
    P. Strange. On important thing to note is that the Dirac differential equations come with
    terms like: g_s * \phi(r) and g_v * V(r). We absorb the coupling constants in the potential.

    Parameters
    ----------
    m : double
      The mass of the nucleon in [MeV].

    kappa : int
      The eigenvalue of the K operator.

    scalar_pot : function
      The scalar potential in units of [MeV].
        
    vector_pot : function
      The vector potential in units of [MeV].
    
    rho_pot : function
      The rho-meson potential in units of [MeV].

    coulomb_pot : function
      The coulomb potential in units of [MeV]

    '''
    
    def __init__(self,m,kappa,scalar_pot,vector_pot,rho_pot,coulomb_pot):
        self.m = m
        self.kappa = kappa
        self.scalar_pot = scalar_pot
        self.vector_pot = vector_pot
        self.rho_pot = rho_pot
        self.coulomb_pot = coulomb_pot
        self.hbarc = 197.326

    def U_RMF(self):
        '''
        The derivative function of u(r), determined from the RMF differential equation.

        Returns
        -------
        U : lambda function
          The first derivative of u(r). Inputs are kappa, r, u, v, and E
         
        '''

        U = lambda r, u, v, E: - (self.kappa / r) * self.hbarc * u + (E + self.m - self.scalar_pot(r) - self.vector_pot(r) - self.rho_pot(r) - self.coulomb_pot(r)) * v

        return U

    def V_RMF(self):
        '''
        The derivative function of v(r), determined from the RMF differential equation.

        Returns
        -------
        V : lambda function
          The first derivative of v(r). Inputs are kappa, r, u, v, and E
         
        '''

        V = lambda r, u, v, E: ((self.kappa / r) * self.hbarc * v - (E - self.m + self.scalar_pot(r) - self.vector_pot(r) - self.rho_pot(r) - self.coulomb_pot(r)) * u)

        return V
    
    def initial_conditions(self, limit = ['zero','inf']):
        '''
        RMF initial conditions for the u and v wavefunctions at either
        r -> 0 or r -> inf

        Parameters
        ----------
        limit : string
          Specifies which initial conditions you want, small or large r.

        Returns
        -------
        u_IC : function
          The initial conditions for the u wavefunction for the chosen limit.
          Is a function of energy

        v_IC : function
          The initial conditions for the v wavefunction for the chosen limit.
          Is a function of energy

        '''

        if limit == 'zero':
            if self.kappa < 0:
                u_IC = lambda r, E: r**(-self.kappa)
                v_IC = lambda r, E : -(E - self.m + self.scalar_pot(r) - self.vector_pot(r) - self.rho_pot(r) - self.coulomb_pot(r)) * r**(-self.kappa + 1) / (1 - 2 * self.kappa) / self.hbarc
            if self.kappa > 0:
                u_IC = lambda r, E: (E + self.m - self.scalar_pot(r) - self.vector_pot(r) - self.rho_pot(r) - self.coulomb_pot(r)) * r**(self.kappa + 1) / (1 + 2 * self.kappa) / self.hbarc
                v_IC = lambda r, E : r**(self.kappa)
        else:
            u_IC = lambda r, E: np.exp( -np.sqrt(self.m**2 - E**2) * r / self.hbarc) 
            v_IC = lambda r, E: np.exp( -np.sqrt(self.m**2 - E**2) * r / self.hbarc) 

        return u_IC, v_IC

def woods_saxon_pot(pot_strength,R0,diffuseness):
    '''
    The Woods-Saxon potential

    Parameters
    ----------
    pot_strength: double
      The potential strength in units of [MeV]

    R0 : double
      The characteristic radius in units of [fm].
    
    diffuseness : double
      The diffuseness in units of [fm]

    Returns
    -------
    ws_pot : function
      The Woods Saxon potential where the function inputs are radius.
    
    '''

    ws_pot = lambda r: pot_strength / ( 1 + np.exp( (np.array(r) - R0) / diffuseness ) )

    return ws_pot