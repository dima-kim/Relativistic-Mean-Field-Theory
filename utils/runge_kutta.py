import numpy as np

class runge_kutta:
    '''
    Numerical Runge-Kutta integrator class. Specifically to solve the following coupled
    first order differential equations:

                                    du(r)/dr = U(r,u(r),v(r))
                                    dv(r)/dr = V(r,u(r),v(r))
    
                                        u(r) = g(r) / r
                                        v(r) = f(r) / r
    
    We are using the same conventions as in the book: "Relativistic Quantum Mechanics" by 
    P. Strange.

    This class is specifically for solving differential equations using forward and backward
    integration.

    Parameters
    ----------
    r_min : double
      The minimum radial position. In units of [fm]

    r_middle : double
      The matching radial position. In units of [fm]

    r_max : double
      The maximum radial position. In units of [fm]
    
    N_steps : int
      The number of steps you want to take to get to r_middle from r_min or r_max,
      depending on whether you do forward or backward integration.

    '''

    def __init__(self, r_min, r_middle, r_max, N_steps):
        self.r_min = r_min
        self.r_middle = r_middle
        self.r_max = r_max
        self.N_steps = N_steps
        self.hbarc = 197.326

    def forward_RK4(self, u_initial, v_initial, U, V, E_guess):
        '''
        Performs forward RK4.

        Parameters
        ----------
        u_initial : double
          The reduced radial wavefunction of g(r), evaluated at r_min.
    
        v_initial : double
          The reduced radial wavefunction of f(r), evaluated at r_min.

        U : function
          The function U, that determines the derivative of u(r), from the 
          differential equation. The units are [MeV] * [u].

        V : function
          The function U, that determines the derivative of v(r), from the 
          differential equation. The units are [MeV] * [u].
        
        E_guess: double
          The energy guess. In units of [MeV].

        Returns
        -------
        r_array : 1-D ndarray
          An array of the radial positions the RK4 evaluated at. In units of [fm].

        u_array : 1-D ndarray
          An array of the reduced radial waveunction of g(r), evaluated on r_array. 
          In units of [fm^-1/2].
        
        v_array : 1-D ndarray
          An array of the reduced radial waveunction of f(r), evaluated on r_array.
          In units of [fm^-1/2].
        
        '''
        
        # Initialize arrays.
        r_array = np.zeros(self.N_steps)
        u_array = np.zeros(self.N_steps)
        v_array = np.zeros(self.N_steps)

        # Set initial conditions 
        r_array[0] = self.r_min
        u_array[0] = u_initial
        v_array[0] = v_initial

        step_size = (self.r_middle - self.r_min) / (self.N_steps-1)

        for i in range(self.N_steps-1):
            U1 = U(r_array[i], u_array[i], v_array[i], E_guess)
            V1 = V(r_array[i], u_array[i], v_array[i], E_guess)

            U2 = U(r_array[i] + ( step_size / 2. ), u_array[i] + U1 * ( step_size / 2. ) / self.hbarc, v_array[i] + V1 * ( step_size / 2. ) / self.hbarc, E_guess)
            V2 = V(r_array[i] + ( step_size / 2. ), u_array[i] + U1 * ( step_size / 2. ) / self.hbarc, v_array[i] + V1 * ( step_size / 2. ) / self.hbarc, E_guess)

            U3 = U(r_array[i] + ( step_size / 2. ), u_array[i] + U2 * ( step_size / 2. ) / self.hbarc, v_array[i] + V2 * ( step_size / 2. ) / self.hbarc, E_guess)
            V3 = V(r_array[i] + ( step_size / 2. ), u_array[i] + U2 * ( step_size / 2. ) / self.hbarc, v_array[i] + V2 * ( step_size / 2. ) / self.hbarc, E_guess)
            
            U4 = U(r_array[i] + step_size, u_array[i] + U3 * step_size / self.hbarc, v_array[i] + V3 * step_size / self.hbarc, E_guess)
            V4 = V(r_array[i] + step_size, u_array[i] + U3 * step_size / self.hbarc, v_array[i] + V3 * step_size / self.hbarc, E_guess)
    
            u_array[i + 1] = u_array[i] + ( step_size / 6. ) * (U1 + 2. * U2 + 2. * U3 + U4) / self.hbarc
            v_array[i + 1] = v_array[i] + ( step_size / 6. ) * (V1 + 2. * V2 + 2. * V3 + V4) / self.hbarc
            r_array[i + 1] = r_array[i] + step_size

        return r_array, u_array, v_array

    def backward_RK4(self, u_initial, v_initial, U, V, E_guess):
        '''
        Performs backward RK4.

        Parameters
        ----------
        u_initial : double
          The reduced radial wavefunction of g(r), evaluated at r_min.
    
        v_initial : double
          The reduced radial wavefunction of f(r), evaluated at r_min.

        U : function
          The function U, that determines the derivative of u(r), from the 
          differential equation. The units are [MeV] * [u].

        V : function
          The function U, that determines the derivative of v(r), from the 
          differential equation. The units are [MeV] * [u].
        
        E_guess: double
          The energy guess. In units of [MeV].

        Returns
        -------
        r_array : 1-D ndarray
          An array of the radial positions the RK4 evaluated at. In units of [fm].

        u_array : 1-D ndarray
          An array of the reduced radial waveunction of g(r), evaluated on r_array. 
          In units of [fm^-1/2].
        
        v_array : 1-D ndarray
          An array of the reduced radial waveunction of f(r), evaluated on r_array.
          In units of [fm^-1/2].
        
        '''
        
        # Initialize arrays
        r_array = np.zeros(self.N_steps)
        u_array = np.zeros(self.N_steps)
        v_array = np.zeros(self.N_steps)

        # Set initial conditions 
        r_array[0] = self.r_max
        u_array[0] = u_initial
        v_array[0] = v_initial

        step_size = (self.r_max - self.r_middle) / (self.N_steps - 1)

        for i in range(self.N_steps-1):
            U1 = U(r_array[i], u_array[i], v_array[i], E_guess)
            V1 = V(r_array[i], u_array[i], v_array[i], E_guess)

            U2 = U(r_array[i] - ( step_size / 2. ), u_array[i] - U1 * ( step_size / 2. ) / self.hbarc, v_array[i] - V1 * ( step_size / 2. ) / self.hbarc, E_guess)
            V2 = V(r_array[i] - ( step_size / 2. ), u_array[i] - U1 * ( step_size / 2. ) / self.hbarc, v_array[i] - V1 * ( step_size / 2. ) / self.hbarc, E_guess)

            U3 = U(r_array[i] - ( step_size / 2. ), u_array[i] - U2 * ( step_size / 2. ) / self.hbarc, v_array[i] - V2 * ( step_size / 2. ) / self.hbarc, E_guess)
            V3 = V(r_array[i] - ( step_size / 2. ), u_array[i] - U2 * ( step_size / 2. ) / self.hbarc, v_array[i] - V2 * ( step_size / 2. ) / self.hbarc, E_guess)
            
            U4 = U(r_array[i] - step_size, u_array[i] - U3 * step_size / self.hbarc, v_array[i] - V3 * step_size / self.hbarc, E_guess)
            V4 = V(r_array[i] - step_size, u_array[i] - U3 * step_size / self.hbarc, v_array[i] - V3 * step_size / self.hbarc, E_guess)
    
            u_array[i + 1] = u_array[i] - ( step_size / 6. ) * (U1 + 2. * U2 + 2. * U3 + U4) / self.hbarc
            v_array[i + 1] = v_array[i] - ( step_size / 6. ) * (V1 + 2. * V2 + 2. * V3 + V4) / self.hbarc
            r_array[i + 1] = r_array[i] - step_size

        # Reverse ordering of arrays, so values go from increasing r values
        r_array = r_array[::-1]
        u_array = u_array[::-1]
        v_array = v_array[::-1]

        return r_array, u_array, v_array
