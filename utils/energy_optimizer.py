from utils.runge_kutta import runge_kutta as RK_class
from scipy import integrate
import numpy as np
import scipy as sypy

class energy_optimizer:
    '''
    Class that handles the determination of energy eigenvalues for a given 
    first-order coupled differential equation using RK4.

    Parameters
    ----------
    r_min : double
      The minimum radial distance we will numerically integrate to. In units of [fm].

    r_middle : double
      The radial distance we will integrate forwards and backwards to. In units of [fm].

    r_max : double
      The maximum radial distance we will numerically integrate to. In units of [fm].
    
    N_steps : int
      The number of steps we will numerically integrate.
    
    U : function
      The function U, that determines the derivative of u(r), from the 
      differential equation

    V : function
      The function U, that determines the derivative of v(r), from the 
      differential equation
    
    '''

    def __init__(self,r_min,r_middle,r_max,N_steps,U,V):
        self.r_min = r_min
        self.r_middle = r_middle
        self.r_max = r_max
        self.N_steps = N_steps
        self.U = U
        self.V = V    
        # self.hbarc = 197.326
        self.hbarc = 197.33

    def RK4_dirac_energy_guess(self, E_guess, diff_eq_class):
        '''
        Uses RK4 to integrate in and out the Dirac Differential Equation.

        Parameters
        ----------
        E_guess : double
          The energy guess. In units of [MeV]

        diff_eq_class : class
          The differential equation class.

        Returns
        -------
        [u_difference, v_boundary] : double
          The discontinuity in the u(r) wavefunction and the value of the v wavefunction 
          at the boundary.
        
        r_array : 1-D ndarray
          An array of the radial positions the RK4 evaluated at. In units of [fm].

        u_array : 1-D ndarray
          An array of the reduced radial waveunction of g(r), evaluated on r_array. 
          In units of [fm^-1/2].
        
        v_array : 1-D ndarray
          An array of the reduced radial waveunction of f(r), evaluated on r_array.
          In units of [fm^-1/2].

        '''

        u_initial_L, v_initial_L = diff_eq_class.initial_conditions(limit='zero')
        u_initial_R, v_initial_R = diff_eq_class.initial_conditions(limit='inf')

        RK4_integrator = RK_class(self.r_min,self.r_middle,self.r_max,self.N_steps)

        r_array_L, u_array_L, v_array_L = RK4_integrator.forward_RK4(u_initial_L(self.r_min,E_guess),v_initial_L(self.r_min,E_guess),self.U,self.V,E_guess)
        r_array_R, u_array_R, v_array_R = RK4_integrator.backward_RK4(u_initial_R(self.r_max,E_guess),v_initial_R(self.r_max,E_guess),self.U,self.V,E_guess)

        # Convetionally calculate a scale factor to make v(r) continuous at the boundary
        scale_factor = v_array_L[-1] / v_array_R[0]

        # Rescale the entire right fields
        u_array_R = scale_factor * u_array_R
        v_array_R = scale_factor * v_array_R 

        # Normalize the whole wavefunction
        norm = 1 / np.sqrt(integrate.simpson(u_array_L * u_array_L + v_array_L * v_array_L, r_array_L) + 
                           integrate.simpson(u_array_R * u_array_R + v_array_R * v_array_R, r_array_R))
        
        u_array_L = norm * u_array_L
        v_array_L = norm * v_array_L
        u_array_R = norm * u_array_R
        v_array_R = norm * v_array_R

        u_difference = u_array_R[0] - u_array_L[-1]

        # Concatenate the left and the right solutions take the value at
        # r_middle to be the average of the left and right RK4 solutions
        u_array_R[0] = 0.5 * (u_array_L[-1] + u_array_R[0])
        v_array_R[0] = 0.5 * (v_array_L[-1] + v_array_R[0])

        r_array = np.concatenate((r_array_L[:-1],r_array_R))
        u_array = np.concatenate((u_array_L[:-1],u_array_R))
        v_array = np.concatenate((v_array_L[:-1],v_array_R))

        return [u_difference, v_array_R[0]], r_array, u_array, v_array
    
    def find_energy_eigenvalue(self, E_guess, diff_eq_class, tol = 5E-3):
        '''
        Determine the energy eigenvalue using an iterative method that
        can be derived using the differential equation. Can be found on page 41 in:

                    https://www3.nd.edu/~johnson/Class01F/chap2a.pdf

        Parameters
        ----------
        E_guess : double
          The initial energy guess in units of [MeV].
        
        diff_eq_class : class
          The differential equation class.
        
        tol : double
          The tolerance in change in energy, tells the algorithm when to stop

        Returns
        -------
        E : double
          The energy solution in units of [MeV].
        
        r_array : 1-D ndarray
          The radial positions in units of [fm].
        
        u_array : 1-D ndarray
          The u wavefunction in units of [fm^-1/2].

        w_array : 1-D ndarray
          The w wavefunction in units of [fm^-1/2].
    
        '''

        E = E_guess
        deltaE = -1

        # The energy increment must be modified depending on your problem.
        # For coulomb potential 0.01 works well.
        energy_increment = 0.1

        # We want deltaE to be moving towards the solution. Essentially we want to 
        # sweep the energies from the bottom to get all of the energy eigenvalues.
        while deltaE < 0:
            E = E + energy_increment
            deltaE_variables, r_array, u_array, v_array = self.RK4_dirac_energy_guess(E,diff_eq_class)
            deltaE = deltaE_variables[1] * deltaE_variables[0] * self.hbarc 
        
        for i in range(50):
            deltaE_variables, r_array, u_array, v_array = self.RK4_dirac_energy_guess(E,diff_eq_class)
            deltaE = deltaE_variables[1] * deltaE_variables[0] * self.hbarc 
            E = E + deltaE 

            if np.abs(deltaE) < tol:
                break
        
        if i == 49:
            print('Did not converge in 50 iterations')
            return 1
        else:
            print('Succesfully converged in ' + str(i) + ' iterations, energy is : ' + str(E))
            print('The discontinuity in the u-wavefunction is: ' + str(deltaE_variables[0]))
            return E, r_array, u_array, v_array, i
        
        


