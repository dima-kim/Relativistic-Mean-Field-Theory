from utils.differential_equation_factory import coulomb_differential_equation, RMF_differential_equation, woods_saxon_pot
from utils.energy_optimizer import energy_optimizer
import numpy as np
from scipy import integrate, interpolate

def coulomb_dirac_energy_solver(input_file):

    # Get the generic parameters
    nucleon_mass = input_file['Sheet1']['NUCLEON MASS [MEV]'][0]

    r_min = input_file['Sheet1']['R_MIN [FM]'][0]
    r_max = input_file['Sheet1']['R_MAX [FM]'][0]
    N_steps = input_file['Sheet1']['N_STEPS'][0]

    xi = input_file['Sheet2']['POTENTIAL STRENGTH'][0]
        
    kappa_list = input_file['Sheet3']['KAPPA']
    r_middle_list = input_file['Sheet3']['MATCH RADIUS [FM]']
    energy_guess = input_file['Sheet3']['ENERGY GUESS [MEV]'][0]

    energy_sol = []
    wavefunction_sol = []
    prev_kappa = 0
    for index,kappa in enumerate(kappa_list):
        
        if kappa == prev_kappa:
            energy_guess = energy_guess + 0.08
        
        diff_eq = coulomb_differential_equation(nucleon_mass,xi,kappa)

        U = diff_eq.U_coulomb()
        V = diff_eq.V_coulomb()

        energy_opt = energy_optimizer(r_min,r_middle_list[index],r_max,N_steps,U,V)
        energy, r_array, u_array, v_array = energy_opt.find_energy_eigenvalue(energy_guess,diff_eq)

        energy_sol.append(energy)
        wavefunction_sol.append([r_array,u_array,v_array])
        
        # Use the energy solution as the guess for the next energy
        # We have an energy offset for the case of degeneracies
        energy_offset = 0.03
        energy_guess = energy - energy_offset

        # Take note of the kappa value
        prev_kappa = kappa

    return energy_sol, wavefunction_sol

def RMF_dirac_energy_solver(input_file,scalar_pot,vector_pot,isospin):
    '''
    Solve for the energy eigenvalues and wavefunctions for a specific
    nucleus, whose information is given in the input file.

    Parameters
    ----------
    input_file : string
      The address of the excel file that contains all the relevant 
      parameters for the nucleus we want to calculate the single
      particle wavefunctions for.
    
    Returns
    -------
    energy_sol : 2-D ndarray
      An array that contains the energy eigenvalues and kappa value
      we obtained the energy for.

    wavefunction_sol : 3-D ndarray
      A 3-D array that contains r_array, u_array, and v_array for each
      energy eigenvalue we calculated. 

    '''

    # Get the generic parameters
    nucleon_mass = input_file['Sheet1']['NUCLEON MASS [MEV]'][0]

    r_min = input_file['Sheet1']['R_MIN [FM]'][0]
    r_max = input_file['Sheet1']['R_MAX [FM]'][0]
    N_steps = input_file['Sheet1']['N_STEPS'][0]
    
    # In a more detailed study, isospin will also change the potentials 
    # we are working with, this will simply just pick out different
    # potentials parameters and potential constructions in this function
    # no need to modify any other existing functions to include isospin.
    if isospin == 1:
        kappa_list = input_file['Sheet3']['KAPPA']
        angular_mom_list = input_file['Sheet3']['2J+1']
        r_middle_list = input_file['Sheet3']['MATCH RADIUS [FM]']
        energy_guess = input_file['Sheet3']['ENERGY GUESS [MEV]'][0]
    else:
        kappa_list = input_file['Sheet4']['KAPPA']
        angular_mom_list = input_file['Sheet4']['2J+1']
        r_middle_list = input_file['Sheet4']['MATCH RADIUS [FM]']
        energy_guess = input_file['Sheet4']['ENERGY GUESS [MEV]'][0]

    energy_sol = []
    wavefunction_sol = []
    for index,kappa in enumerate(kappa_list):

        diff_eq = RMF_differential_equation(nucleon_mass,kappa,scalar_pot,vector_pot)
        U = diff_eq.U_RMF()
        V = diff_eq.V_RMF()
        energy_opt = energy_optimizer(r_min,r_middle_list[index],r_max,N_steps,U,V)

        energy, r_array, u_array, v_array = energy_opt.find_energy_eigenvalue(energy_guess,diff_eq)
        energy_sol.append([energy,angular_mom_list[index],isospin])
        wavefunction_sol.append([r_array,u_array,v_array])

        # Set the next energy guess to be the solution we found.
        # Also shift it backwards by 1.5 MeV so that we dont miss solutions
        # that have energies close to one another.
        energy_guess = energy - 1.5

    return np.array(energy_sol), np.array(wavefunction_sol)
    
def generate_densities(energy_sol, wavefunction_sol):
    '''
    Generate the scalar and baryon densitites from the single paricle solutions
    to the Dirac Equation.

    Parameters
    ----------
    energy_sol : 2-D ndarray
      An array that contains the energy eigenvalues and kappa value
      we obtained the energy for.

    wavefunction_sol : 3-D ndarray
      A 3-D array that contains r_array, u_array, and v_array for each
      energy eigenvalue we calculated. 

    Returns
    -------
    rho_b_array : 1-D ndarray
      The baryon density, in units of [fm^-3]
    
    rho_s_array : 1-D ndarray
      The scalar density, in units of [fm^-3]

    '''

    # For now, we don't include electromagnetic effects, thus
    # neutron and proton are the same
    dim = len(energy_sol)

    # First we need to calculate the square of the wavefunctions.
    # The formulae for the densities requires the square of the 
    # wavefunctions and radial positions. Squaring the wavefunction_sol
    # array will give us all of this.
    wfk_squared = wavefunction_sol * wavefunction_sol

    rho_b_array = 0
    rho_s_array = 0
    for i in range(dim):
        rho_b_array = rho_b_array + (energy_sol[i][1] / (4 * np.pi * wfk_squared[i][0])) * (wfk_squared[i][1] + wfk_squared[i][2])
        rho_s_array = rho_s_array + (energy_sol[i][1] / (4 * np.pi * wfk_squared[i][0])) * (wfk_squared[i][1] - wfk_squared[i][2])

    return rho_b_array, rho_s_array

def generate_I1(r_array,rho_array,m_meson):
    '''
    Generate the I1 integral needed for solving the meson differential equations, presented in page 140 of:

         Computational Nuclear Physics 1: Nuclear Structure by K. Langanke J. A. Maruhn S. E. Koonin   
    
    Parameters
    ----------
    r_array : 1-D ndarray
      An array of the radial positions everything is evaluated at. In units of [fm].
    
    rho_array : 1-D ndarray
      The scalar or baryon density. In units of [fm^-3].
    
    m_meson : double 
      The mass of the meson. In units of [MeV]

    Returns
    -------
    I1_array : 1-D ndarray
      The I1 integral evaluated at the radial positions given by r_array. In units of [fm^-1].
    
    '''

    hbarc = 197.326
    I1_array = []
    for index in range(len(r_array)):
        
        r_integrand = r_array[0:index+1]
        rho_integrand = rho_array[0:index+1]
        exp_integrand = np.exp(m_meson * r_integrand / hbarc)

        I1_array.append(integrate.simpson( r_integrand * rho_integrand * exp_integrand, r_integrand ))

    return np.array(I1_array)

def generate_I2(r_array,rho_array,m_meson):
    '''
    Generate the I2 integral needed for solving the meson differential equations, presented in page 140 of:

         Computational Nuclear Physics 1: Nuclear Structure by K. Langanke J. A. Maruhn S. E. Koonin   
    
    Parameters
    ----------
    r_array : 1-D ndarray
      An array of the radial positions everything is evaluated at. In units of [fm].
    
    rho_array : 1-D ndarray
      The scalar or baryon density. In units of [fm^-3].
    
    m_meson : double 
      The mass of the meson. In units of [MeV]

    Returns
    -------
    I2_array : 1-D ndarray
      The I2 integral evaluated at the radial positions given by r_array. In units of [fm^-1].
    
    '''

    hbarc = 197.326
    I2_array = []
    array_dim = len(r_array)
    for index in range(array_dim):
        
        r_integrand = r_array[index:array_dim]
        rho_integrand = rho_array[index:array_dim]
        exp_integrand = np.exp(-m_meson * r_integrand / hbarc)

        I2_array.append(integrate.simpson( r_integrand * rho_integrand * exp_integrand, r_integrand ))

    return np.array(I2_array)

def generate_potential(r_array,rho_array,m_meson,meson_coupling):
    '''
    Solves the meson field differential equations by Green's function method. More information in
    
     Computational Nuclear Physics 1: Nuclear Structure by K. Langanke J. A. Maruhn S. E. Koonin   

    Paramters
    ---------
    r_array : 1-D ndarray
      An array of the radial positions everything is evaluated at. In units of [fm].
    
    rho_array : 1-D ndarray
      The scalar or baryon density. In units of [fm^-3].
    
    m_meson : double 
      The mass of the meson. In units of [MeV]
    
    meson_coupling : double
      The meson coupling constant with the nucleon.
    
    Returns
    -------
    pot_interp : interpolated function
      The interpolated meson 'potential' field in units of [MeV].

    '''

    hbarc = 197.326
    I1 = generate_I1(r_array,rho_array,m_meson)
    I2 = generate_I2(r_array,rho_array,m_meson)

    # Remember that are including the factor of the coupling into our definition of the potential.
    pot_array = hbarc**2 * (meson_coupling**2 / (2 * m_meson * r_array)) * ( np.exp(-m_meson * r_array / hbarc) * (I1 - I2[0]) + np.exp(m_meson * r_array / hbarc) * I2)

    pot_interp = interpolate.interp1d(r_array, pot_array)

    return pot_interp
                     
def solve_SCRMFT(input_file, tol = 1E-6):
    '''
    Solves the full self-consistent relativistic mean field theory.

    Parameters
    ----------
    input_file : string
      The address of the excel file that contains all the relevant 
      parameters for the nucleus we want to calculate the single
      particle wavefunctions for.

    Returns
    -------
    total_energy_sol : 2-D ndarray
      The single particle energies of the proton and neutron in units of [MeV].
    
    total_wavefunction_sol : 2-D ndarray
      The single particle wavefunction solutions of the proton and neutron,
      in units of [fm^-1/2].
      
    '''
    
    # Meson Parameters
    m_scalar = input_file['Sheet1']['SIGMA MASS [MEV]'][0]
    m_vector = input_file['Sheet1']['OMEGA MASS [MEV]'][0]
    gs = input_file['Sheet2']['SCALAR COUPLING'][0]
    gv = input_file['Sheet2']['VECTOR COUPLING'][0]

    # First iteration involves the Woods-Saxon potential
    scalar_pot_strength = input_file['Sheet2']['SCALAR STRENGTH [MEV]'][0]
    vector_pot_strength = input_file['Sheet2']['VECTOR STRENGTH [MEV]'][0]
    R0 = input_file['Sheet2']['R0 [FM]'][0]
    diffuseness = input_file['Sheet2']['DIFFUSENESS [FM]'][0]

    scalar_pot = woods_saxon_pot(scalar_pot_strength,R0,diffuseness)
    vector_pot = woods_saxon_pot(vector_pot_strength,R0,diffuseness)

    # Initialize the difference to ensure while loop is true in first iteration
    scalar_pot_diff = 100
    vector_pot_diff = 100
    iteration = 0

    while scalar_pot_diff > tol or vector_pot_diff > tol:

        # Currently our potentials don't distinguish between proton and neutron
        p_energy_sol, p_wavefunction_sol = RMF_dirac_energy_solver(input_file,scalar_pot,vector_pot,isospin=1)
        n_energy_sol, n_wavefunction_sol = p_energy_sol, p_wavefunction_sol

        total_energy_sol = np.concatenate([p_energy_sol,n_energy_sol])
        total_wavefunction_sol = np.concatenate([p_wavefunction_sol,n_wavefunction_sol])

        r_array = total_wavefunction_sol[0][0]

        rho_b, rho_s = generate_densities(total_energy_sol,total_wavefunction_sol)

        # Previous potential values
        scalar_pot_before = scalar_pot(r_array)
        vector_pot_before = vector_pot(r_array)

        scalar_pot = generate_potential(r_array, rho_s, m_scalar, gs)
        vector_pot = generate_potential(r_array, rho_b, m_vector, gv)

        # New potential values
        scalar_pot_after = scalar_pot(r_array)
        vector_pot_after = vector_pot(r_array)

        scalar_pot_diff = np.abs(np.max(scalar_pot_before - scalar_pot_after))
        vector_pot_diff = np.abs(np.max(vector_pot_before - vector_pot_after))

        iteration = iteration + 1

    print('Self Consistency Acheived in ' + str(iteration) + ' iterations')

    return total_energy_sol, total_wavefunction_sol


    

    


