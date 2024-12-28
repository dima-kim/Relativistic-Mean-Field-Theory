from utils.differential_equation_factory import coulomb_differential_equation, RMF_differential_equation, woods_saxon_pot
from utils.energy_optimizer import energy_optimizer
import numpy as np
import pandas as pd
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

def RMF_dirac_energy_solver(input_file,scalar_pot,vector_pot,rho_pot,coulomb_pot,isospin):
    '''
    Solve for the energy eigenvalues and wavefunctions for a specific
    nucleus, whose information is given in the input file.

    Parameters
    ----------
    input_file : string
      The address of the excel file that contains all the relevant 
      parameters for the nucleus we want to calculate the single
      particle wavefunctions for.
    
    scalar_pot : lambda function
      The scalar potential as a function of r. In units of [MeV]

    vector_pot : lambda function
      The vector potential as a function of r. In units of [MeV]

    rho_pot : lambda function
      The rho potential as a function of r. In units of [MeV]

    coulomb_pot : lambda function
      The coulomb potential as a function of r. In units of [MeV]
    
    isospin : double 
      The isospin quatnum number of the nucleon. Either -1/2 or 1/2 for 
      proton or neutron respectively. 

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
    
    if isospin == 1/2:
        kappa_list = input_file['Sheet3']['KAPPA']
        angular_mom_list = input_file['Sheet3']['2J+1']
        r_middle_list = input_file['Sheet3']['MATCH RADIUS [FM]']
        energy_guess = input_file['Sheet3']['ENERGY GUESS [MEV]'][0]
        state_list = input_file['Sheet3']['LABEL']
    else:
        kappa_list = input_file['Sheet4']['KAPPA']
        angular_mom_list = input_file['Sheet4']['2J+1']
        r_middle_list = input_file['Sheet4']['MATCH RADIUS [FM]']
        energy_guess = input_file['Sheet4']['ENERGY GUESS [MEV]'][0]
        state_list = input_file['Sheet4']['LABEL']

    energy_sol = []
    wavefunction_sol = []
    for index,kappa in enumerate(kappa_list):
        
        # Including the factor of the isospin the the rho potential present in the Dirac equation.
        rho_pot_isospin = lambda r: isospin * rho_pot(r)
        coulomb_pot_isospin = lambda r: (isospin + 0.5) * coulomb_pot(r)

        diff_eq = RMF_differential_equation(nucleon_mass,kappa,scalar_pot,vector_pot,rho_pot_isospin,coulomb_pot_isospin)
        U = diff_eq.U_RMF()
        V = diff_eq.V_RMF()
        energy_opt = energy_optimizer(r_min,r_middle_list[index],r_max,N_steps,U,V)

        print('----------------------------------------------------------------------------')
        print(state_list[index])
        energy, r_array, u_array, v_array, iterations = energy_opt.find_energy_eigenvalue(energy_guess,diff_eq)

        # I believe that I am going back enough to capture the lowest energy and my energy resolution is fine enough.
        # Furthermore, I have set the match radii so that they are away from nodes. If the iteration is 0
        # it must be that the energy guess makes one of the wavefunctions 0 over a range of r-values. Thus we will continue to
        # offset the energy guess until we get a result.
        while iterations == 0:
            print("########## SOLVED IN 0 ITERATIONS... LETS RE-DO WITH A SLIGHTLY LARGER ENERGY GUESS ##########")
            energy_guess_increment = 1
            energy_guess = energy_guess + energy_guess_increment
            energy, r_array, u_array, v_array, iterations = energy_opt.find_energy_eigenvalue(energy_guess,diff_eq)

        energy_sol.append([energy,angular_mom_list[index],isospin])
        wavefunction_sol.append([r_array,u_array,v_array])

        # Set the next energy guess to be the solution we found.
        # Also shift it backwards by 3 MeV so that we dont miss solutions
        # that have energies close to one another.
        energy_guess_increment = 3
        energy_guess = energy - energy_guess_increment

    return np.array(energy_sol), np.array(wavefunction_sol)
    
def generate_densities(energy_sol, wavefunction_sol):
    '''
    Generate the scalar, baryon, '3', and proton densitites from the single paricle solutions
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

    rho_3_array : 1-D ndarray
      The density for the rho-meson differential equation, in units of [fm^-3]

    rho_p_array : 1-D ndarray
      The proton density, in units of [fm^-3]

    '''

    dim = len(energy_sol)

    # First we need to calculate the square of the wavefunctions.
    # The formulae for the densities requires the square of the 
    # wavefunctions and radial positions. Squaring the wavefunction_sol
    # array will give us all of this.
    wfk_squared = wavefunction_sol * wavefunction_sol

    rho_b_array = 0
    rho_s_array = 0
    rho_3_array = 0
    rho_p_array = 0
    for i in range(dim):
        rho_b_array = rho_b_array + (energy_sol[i][1] / (4 * np.pi * wfk_squared[i][0])) * (wfk_squared[i][1] + wfk_squared[i][2])
        rho_s_array = rho_s_array + (energy_sol[i][1] / (4 * np.pi * wfk_squared[i][0])) * (wfk_squared[i][1] - wfk_squared[i][2])
        rho_3_array = rho_3_array + 0.5 * (energy_sol[i][1] / (4 * np.pi * wfk_squared[i][0])) * (wfk_squared[i][1] + wfk_squared[i][2]) * (-1)**(energy_sol[i][2] - 0.5)
        rho_p_array = rho_p_array + (energy_sol[i][1] / (4 * np.pi * wfk_squared[i][0])) * (wfk_squared[i][1] + wfk_squared[i][2]) * (energy_sol[i][2] + 0.5)

    return rho_b_array, rho_s_array, rho_3_array, rho_p_array

def generate_I1_meson(r_array,rho_array,m_meson):
    '''
    Generate the I1 integral needed for solving the meson differential equations, presented in page 140 of:

         Computational Nuclear Physics 1: Nuclear Structure by K. Langanke J. A. Maruhn S. E. Koonin   
    
    Parameters
    ----------
    r_array : 1-D ndarray
      An array of the radial positions everything is evaluated at. In units of [fm].
    
    rho_array : 1-D ndarray
      The scalar, baryon, or '3' density. In units of [fm^-3].
    
    m_meson : double 
      The mass of the meson. In units of [MeV]

    Returns
    -------
    I1_array : 1-D ndarray
      The I1 integral evaluated at the radial positions given by r_array. In units of [fm^-1].
    
    '''

    hbarc = 197.33
    I1_array = []
    for index in range(len(r_array)):
        
        r_integrand = r_array[0:index+1]
        rho_integrand = rho_array[0:index+1]
        exp_integrand = np.exp(m_meson * r_integrand / hbarc)

        I1_array.append(integrate.simpson( r_integrand * rho_integrand * exp_integrand, r_integrand ))

    return np.array(I1_array)

def generate_I2_meson(r_array,rho_array,m_meson):
    '''
    Generate the I2 integral needed for solving the meson differential equations, presented in page 140 of:

         Computational Nuclear Physics 1: Nuclear Structure by K. Langanke J. A. Maruhn S. E. Koonin   
    
    Parameters
    ----------
    r_array : 1-D ndarray
      An array of the radial positions everything is evaluated at. In units of [fm].
    
    rho_array : 1-D ndarray
      The scalar, baryon, or '3' density. In units of [fm^-3].
    
    m_meson : double 
      The mass of the meson. In units of [MeV]

    Returns
    -------
    I2_array : 1-D ndarray
      The I2 integral evaluated at the radial positions given by r_array. In units of [fm^-1].
    
    '''

    hbarc = 197.33
    I2_array = []
    array_dim = len(r_array)
    for index in range(array_dim):
        
        r_integrand = r_array[index:array_dim]
        rho_integrand = rho_array[index:array_dim]
        exp_integrand = np.exp(-m_meson * r_integrand / hbarc)

        I2_array.append(integrate.simpson( r_integrand * rho_integrand * exp_integrand, r_integrand ))

    return np.array(I2_array)

def generate_meson_potential(r_array,rho_array,m_meson,meson_coupling):
    '''
    Solves the meson field differential equations by Green's function method and creates the meson potential
    by multiplying the meson field by the coupling. More information in:
    
     Computational Nuclear Physics 1: Nuclear Structure by K. Langanke J. A. Maruhn S. E. Koonin   

    Paramters
    ---------
    r_array : 1-D ndarray
      An array of the radial positions everything is evaluated at. In units of [fm].
    
    rho_array : 1-D ndarray
      The scalar, baryon, or '3' density. In units of [fm^-3].
    
    m_meson : double 
      The mass of the meson. In units of [MeV]
    
    meson_coupling : double
      The meson coupling constant with the nucleon.
    
    Returns
    -------
    pot_interp : interpolated function
      The interpolated meson 'potential' field in units of [MeV].

    '''

    hbarc = 197.33
    I1 = generate_I1_meson(r_array,rho_array,m_meson)
    I2 = generate_I2_meson(r_array,rho_array,m_meson)

    # Remember that we are generating the meson POTENTIAL, thus we multiply
    # the meson field by an extra factor of the couping.
    pot_array = hbarc**2 * (meson_coupling**2 / (2 * m_meson * r_array)) * ( np.exp(-m_meson * r_array / hbarc) * (I1 - I2[0]) + np.exp(m_meson * r_array / hbarc) * I2)

    # The reason I use this interpolatior instead of cubic is because in flat regions the 
    # cubic spline can be oscilating. PChipInterpolator tries to keep the shape
    # implied by the data. The only offset is that cubic spline is twice differentiable
    # while PChipInterpolator is once differentiable. This is ok because we only need to differentiate
    # our potentials once!
    pot_interp = interpolate.PchipInterpolator(r_array, pot_array)

    return pot_interp

def generate_I1_coulomb(r_array,rho_array):
    '''
    Generate the I1 integral needed for solving the coulomb differential equations.
    
    Parameters
    ----------
    r_array : 1-D ndarray
      An array of the radial positions everything is evaluated at. In units of [fm].
    
    rho_array : 1-D ndarray
      The proton density. In units of [fm^-3].

    Returns
    -------
    I1_array : 1-D ndarray
      The I1 integral evaluated at the radial positions given by r_array. In units of [fm^-1].
    
    '''

    I1_array = []
    for index in range(len(r_array)):
        
        r_integrand = r_array[0:index+1]
        rho_integrand = rho_array[0:index+1]

        I1_array.append(integrate.simpson( r_integrand * r_integrand * rho_integrand, r_integrand ))

    return np.array(I1_array)

def generate_I2_coulomb(r_array,rho_array):
    '''
    Generate the I2 integral needed for solving the coulomb differential equations.
    
    Parameters
    ----------
    r_array : 1-D ndarray
      An array of the radial positions everything is evaluated at. In units of [fm].
    
    rho_array : 1-D ndarray
      The proton density. In units of [fm^-3].

    Returns
    -------
    I2_array : 1-D ndarray
      The I2 integral evaluated at the radial positions given by r_array. In units of [fm^-1].
    
    '''

    I2_array = []
    array_dim = len(r_array)
    for index in range(array_dim):
        
        r_integrand = r_array[index:array_dim]
        rho_integrand = rho_array[index:array_dim]

        I2_array.append(integrate.simpson( r_integrand * rho_integrand, r_integrand ))

    return np.array(I2_array)

def generate_coulomb_potential(r_array,rho_array,electron_coupling):
    '''
    Solves for the coulomb field by Green's function method, and creates the coulomb
    potential by multiplying it by the coupling constant. 

    Paramters
    ---------
    r_array : 1-D ndarray
      An array of the radial positions everything is evaluated at. In units of [fm].
    
    rho_array : 1-D ndarray
      The proton density. In units of [fm^-3].
    
    electron_coupling : double
      The electron coupling constant with the nucleon.
    
    Returns
    -------
    pot_interp : interpolated function
      The interpolated meson 'potential' field in units of [MeV].

    '''

    hbarc = 197.33
    I1 = generate_I1_coulomb(r_array,rho_array)
    I2 = generate_I2_coulomb(r_array,rho_array)

    # Remember we are generating the coulomb POTENTIAL, thus we multiply the coulomb field
    # by an extra factor of the coupling.
    pot_array = hbarc * electron_coupling**2 * ( (I1 / r_array) + I2)

    # The reason I use this interpolatior instead of cubic is because in flat regions the 
    # cubic spline can be oscilating. PChipInterpolator tries to keep the shape
    # implied by the data. The only offset is that cubic spline is twice differentiable
    # while PChipInterpolator is once differentiable. This is ok because we only need to differentiate
    # our potentials once!
    pot_interp = interpolate.PchipInterpolator(r_array, pot_array)

    return pot_interp
                     
def solve_SCRMFT(input_file, output_file_name, tol = 5E-3):
    '''
    Solves the full self-consistent relativistic mean field theory.

    Parameters
    ----------
    input_file : string
      The address of the excel file that contains all the relevant 
      parameters for the nucleus we want to calculate the single
      particle wavefunctions for.

    tol : double
      The tolerance for self-consistency condition.

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
    m_rho = input_file['Sheet1']['RHO MASS [MEV]'][0]
    gs = input_file['Sheet2']['SCALAR COUPLING'][0]
    gv = input_file['Sheet2']['VECTOR COUPLING'][0]
    grho = input_file['Sheet2']['RHO COUPLING'][0]

    # Coulomb Parameters
    e = input_file['Sheet2']['COULOMB COUPLING'][0]

    # First iteration involves the Woods-Saxon potential
    scalar_pot_strength = input_file['Sheet2']['SCALAR STRENGTH [MEV]'][0]
    vector_pot_strength = input_file['Sheet2']['VECTOR STRENGTH [MEV]'][0]
    rho_pot_strength = input_file['Sheet2']['RHO STRENGTH [MEV]'][0]
    coulomb_pot_strength = input_file['Sheet2']['COULOMB STRENGTH [MEV]'][0]
    R0 = input_file['Sheet2']['R0 [FM]'][0]
    diffuseness = input_file['Sheet2']['DIFFUSENESS [FM]'][0]

    scalar_pot = woods_saxon_pot(scalar_pot_strength,R0,diffuseness)
    vector_pot = woods_saxon_pot(vector_pot_strength,R0,diffuseness)
    rho_pot = woods_saxon_pot(rho_pot_strength,R0,diffuseness)
    coulomb_pot = woods_saxon_pot(coulomb_pot_strength,R0,diffuseness)

    # Initialize the difference to ensure while loop is true in first iteration
    scalar_field_diff = 10
    vector_field_diff = 10
    rho_field_diff = 10
    coulomb_field_diff = 10
    iteration = 0

    while scalar_field_diff > tol or vector_field_diff > tol or rho_field_diff > tol or coulomb_field_diff > tol:
        print('')
        print('------------ ITERATION ' + str(iteration) + ' ------------')
        
        print('')
        print('--------------')
        print('NEUTRON STATES')
        print('--------------')
        n_energy_sol, n_wavefunction_sol = RMF_dirac_energy_solver(input_file,scalar_pot,vector_pot,rho_pot,coulomb_pot,isospin=-1/2)
        
        print('')
        print('-------------')
        print('PROTON STATES')
        print('-------------')
        p_energy_sol, p_wavefunction_sol = RMF_dirac_energy_solver(input_file,scalar_pot,vector_pot,rho_pot,coulomb_pot,isospin=1/2)

        total_energy_sol = np.concatenate([p_energy_sol,n_energy_sol])
        total_wavefunction_sol = np.concatenate([p_wavefunction_sol,n_wavefunction_sol])
        r_array = total_wavefunction_sol[0][0]

        rho_b, rho_s, rho_3, rho_p = generate_densities(total_energy_sol,total_wavefunction_sol)

        # Previous potential values
        scalar_pot_before = scalar_pot(r_array)
        vector_pot_before = vector_pot(r_array)
        rho_pot_before = rho_pot(r_array)
        coulomb_pot_before = coulomb_pot(r_array)

        scalar_pot = generate_meson_potential(r_array, rho_s, m_scalar, gs)
        vector_pot = generate_meson_potential(r_array, rho_b, m_vector, gv)
        rho_pot = generate_meson_potential(r_array, rho_3, m_rho, grho)
        coulomb_pot = generate_coulomb_potential(r_array, rho_p, e)

        # New potential values
        scalar_pot_after = scalar_pot(r_array)
        vector_pot_after = vector_pot(r_array)
        rho_pot_after = rho_pot(r_array)
        coulomb_pot_after = coulomb_pot(r_array)

        # To get the fields we divide out the coupling from the potentials
        scalar_field_diff = np.abs( np.max( (scalar_pot_before - scalar_pot_after)/gs ) )
        vector_field_diff = np.abs( np.max( (vector_pot_before - vector_pot_after)/gv ) )
        rho_field_diff = np.abs( np.max( (rho_pot_before - rho_pot_after)/grho ) )
        coulomb_field_diff = np.abs( np.max( (coulomb_pot_before - coulomb_pot_after)/e ) )

        iteration = iteration + 1

    print('')
    print('Self Consistency Acheived in ' + str(iteration-1) + ' Iterations')

    scalar_field = lambda r: scalar_pot(r) / gs
    vector_field = lambda r: vector_pot(r) / gv
    rho_field = lambda r: rho_pot(r) / grho
    coulomb_field = lambda r: coulomb_pot(r) / e

    ### ---------- Write the wavefunctions and energies ---------- ###
    state_list_p = input_file['Sheet3']['LABEL']
    state_list_n = input_file['Sheet4']['LABEL']
    kappa_list_p = input_file['Sheet3']['KAPPA']
    kappa_list_n = input_file['Sheet4']['KAPPA']
    with pd.ExcelWriter(output_file_name) as writer:
        
        # Create wavefunction dictionaries
        d_p_wf = {'R [FM]': p_wavefunction_sol[0][0], 'U [FM^-1/2] (' + str(state_list_p[0]) + ')' :p_wavefunction_sol[0][1], 'V [FM^-1/2] (' + str(state_list_p[0]) + ')':p_wavefunction_sol[0][2]} 
        d_n_wf = {'R [FM]': n_wavefunction_sol[0][0], 'U [FM^-1/2] (' + str(state_list_n[0]) + ')' :n_wavefunction_sol[0][1], 'V [FM^-1/2] (' + str(state_list_n[0]) + ')':n_wavefunction_sol[0][2]} 

        for i in range(1,p_wavefunction_sol.shape[0]):
            
            d_p_wf_hold = {'U [FM^-1/2] (' + str(state_list_p[i]) + ')':p_wavefunction_sol[i][1], 'V [FM^-1/2] (' + str(state_list_p[i]) + ')':p_wavefunction_sol[i][2]}
            d_p_wf = d_p_wf | d_p_wf_hold

            d_n_wf_hold = {'U [FM^-1/2] (' + str(state_list_n[i]) + ')':n_wavefunction_sol[i][1], 'V [FM^-1/2] (' + str(state_list_n[i]) + ')':n_wavefunction_sol[i][2]}
            d_n_wf = d_n_wf | d_n_wf_hold

        # Create energy dictionaries
        energy_p = [array[0] for array in p_energy_sol]
        energy_n = [array[0] for array in n_energy_sol]
        number_of_states_p = [array[1] for array in p_energy_sol]
        number_of_states_n = [array[1] for array in n_energy_sol]
        d_p_energy = {'LABEL': state_list_p, '2J+1': number_of_states_p, 'KAPPA' : kappa_list_p, 'ENERGY [MEV]': energy_p}
        d_n_energy = {'LABEL': state_list_n, '2J+1': number_of_states_n, 'KAPPA' : kappa_list_n, 'ENERGY [MEV]': energy_n}
        
        # Create wavefunction and emergy data frames
        df_p_wf = pd.DataFrame(d_p_wf)
        df_p_energy = pd.DataFrame(d_p_energy)
        
        df_n_wf = pd.DataFrame(d_n_wf)
        df_n_energy = pd.DataFrame(d_n_energy)

        # Write the DataFrame to a sheet named after the nucleon
        df_p_wf.to_excel(writer, sheet_name='PROTON', index = False)
        df_p_energy.to_excel(writer, sheet_name='PROTON ENERGY', index = False)

        df_n_wf.to_excel(writer, sheet_name='NEUTRON', index = False)
        df_n_energy.to_excel(writer, sheet_name='NEUTRON ENERGY', index = False)

    return p_energy_sol, p_wavefunction_sol, n_energy_sol, n_wavefunction_sol, scalar_field, vector_field, rho_field, coulomb_field

def solve_woods_saxon_RMFT(input_file):
    '''
    Solves the woods-saxon relativistic mean field theory.

    Parameters
    ----------
    input_file : string
      The address of the excel file that contains all the relevant 
      parameters for the nucleus we want to calculate the single
      particle wavefunctions for.

    tol : double
      The tolerance for self-consistency condition.

    Returns
    -------
    total_energy_sol : 2-D ndarray
      The single particle energies of the proton and neutron in units of [MeV].
    
    total_wavefunction_sol : 2-D ndarray
      The single particle wavefunction solutions of the proton and neutron,
      in units of [fm^-1/2].
      
    '''
    # Meson Parameters
    gs = input_file['Sheet2']['SCALAR COUPLING'][0]
    gv = input_file['Sheet2']['VECTOR COUPLING'][0]
    grho = input_file['Sheet2']['RHO COUPLING'][0]

    # Coulomb Parameters
    e = input_file['Sheet2']['COULOMB COUPLING'][0]

    # First iteration involves the Woods-Saxon potential
    scalar_pot_strength = input_file['Sheet2']['SCALAR STRENGTH [MEV]'][0]
    vector_pot_strength = input_file['Sheet2']['VECTOR STRENGTH [MEV]'][0]
    rho_pot_strength = input_file['Sheet2']['RHO STRENGTH [MEV]'][0]
    coulomb_pot_strength = input_file['Sheet2']['COULOMB STRENGTH [MEV]'][0]
    R0 = input_file['Sheet2']['R0 [FM]'][0]
    diffuseness = input_file['Sheet2']['DIFFUSENESS [FM]'][0]

    scalar_pot = woods_saxon_pot(scalar_pot_strength,R0,diffuseness)
    vector_pot = woods_saxon_pot(vector_pot_strength,R0,diffuseness)
    rho_pot = woods_saxon_pot(rho_pot_strength,R0,diffuseness)
    coulomb_pot = woods_saxon_pot(coulomb_pot_strength,R0,diffuseness)
        
    print('')
    print('--------------')
    print('NEUTRON STATES')
    print('--------------')
    n_energy_sol, n_wavefunction_sol = RMF_dirac_energy_solver(input_file,scalar_pot,vector_pot,rho_pot,coulomb_pot,isospin=-1/2)
        
    print('')
    print('-------------')
    print('PROTON STATES')
    print('-------------')
    p_energy_sol, p_wavefunction_sol = RMF_dirac_energy_solver(input_file,scalar_pot,vector_pot,rho_pot,coulomb_pot,isospin=1/2)

    total_energy_sol = np.concatenate([p_energy_sol,n_energy_sol])
    total_wavefunction_sol = np.concatenate([p_wavefunction_sol,n_wavefunction_sol])

    scalar_field = lambda r: scalar_pot(r) / gs
    vector_field = lambda r: vector_pot(r) / gv
    rho_field = lambda r: rho_pot(r) / grho
    coulomb_field = lambda r: coulomb_pot(r) / e

    return total_energy_sol, total_wavefunction_sol, scalar_field, vector_field, rho_field, coulomb_field
    

    


