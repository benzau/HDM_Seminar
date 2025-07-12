import fenics as fe
import torch 
import matplotlib.pyplot as plt
import numpy as np
import time as t

"""
1D Burger: du/dt + u du/dx - eta d^2u/dx^2 = 0
IC: u(x,0) = u0(x) --> sin(x) to begin with
BC: u(x=0, t) = u(x=1, t)
du/dt (x=0, t) = du/dt (x=1, t) 
"""
def solve_pde (initial_function, time):
    start = t.time()
    fe.set_log_active(False)

    nu = 0.05
    #Define the mesh
    GRID_LENGTH = initial_function.shape[-1]-1  #Grid Length same size as initial condition
    mesh = fe.UnitIntervalMesh(GRID_LENGTH)


    class PeriodicBoundaryCondition(fe.SubDomain):
        #Check if I am on x=0
        def inside(self, x, on_boundary):
            return bool(x[0] < fe.DOLFIN_EPS and x[0] > - fe.DOLFIN_EPS and on_boundary)
        
        #Map right to left boundary
        def map(self, x, y):
            y[0] = x[0] - 1    #When called maps the input function (x) to the left boundary (y)

    pbc = PeriodicBoundaryCondition()

    def boundary(x):
        return x[0] < fe.DOLFIN_EPS or x[0] > 1.0 - fe.DOLFIN_EPS
   

    #Define the function space
    #V = fe.FunctionSpace(mesh, 'P', 1, constrained_domain = pbc)
    V = fe.FunctionSpace(mesh, 'P', 1)
    u0 = fe.Constant(0.0)
    bc = fe.DirichletBC(V, u0, boundary)

    #Initialize Test and Trial functions
    u_trial = fe.Function(V)
    v_test = fe.TestFunction(V)

    #Time Discretization with Implicit Euler and assigning initial_condition to nodes for fenics
    u_old = fe.Function(V)

    dofs = V.dofmap().dofs()
    u_coords = V.tabulate_dof_coordinates().reshape(-1)
    u_nodal = np.zeros(len(dofs))

    #Find the nodal values of the FEniCS Function object that correspond to the coordinates in u_coords
    for j, (x) in enumerate(u_coords):
        idx = np.where(np.isclose(x, mesh.coordinates()[:]))[0][0]
        u_nodal[j] = initial_function[idx]
    u_old.vector().set_local(u_nodal)
    u_old.vector().apply('insert')
    
    TIME_STEP_LENGTH = 1/time#initial_function.shape[-1]

    #Weak Form
    F = u_trial*v_test*fe.dx + TIME_STEP_LENGTH* nu* u_trial.dx(0) * v_test.dx(0) *fe.dx - u_old*v_test*fe.dx
    #Solve the PDE
    MAX_TIME = 1.0
    CURRENT_TIME = 0
    k = 0
    num_time_steps = int(MAX_TIME / TIME_STEP_LENGTH)
    u_solutions = np.zeros((num_time_steps + 1, GRID_LENGTH + 1)) #Saving my solutions

    while CURRENT_TIME <= MAX_TIME: #Solving the PDE and assigning new u values to u_old
        fe.solve(F == 0, u_trial, bc)
        
        assigner = fe.FunctionAssigner(V,V)
        assigner.assign(u_old, u_trial)
        u_array = u_trial.compute_vertex_values(mesh)
        u_solutions [num_time_steps-k, :] = u_array	
        
        k += 1
        CURRENT_TIME += TIME_STEP_LENGTH

    end = t.time()
    print("Computing time Fenics: ", end-start)
    # Creating the heat map
    plt.imshow(u_solutions[1:,:],cmap='jet', extent=[0, 1, 0, MAX_TIME])
    plt.colorbar(label='u(x, t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Heat Equation Solution Heatmap')
    filename = 'heat_eq_heatmap'
    plt.savefig(filename)
    #plt.show()
    #plt.close()


    
    u_torch = torch.from_numpy(u_solutions)
    u_torch = u_torch[1:]
    #torch.save(u_torch, 'heat_fenics_sol_tensor_0.01') #Uncomment and save with different nus if wanted
    return u_torch
    
time = 1000 #Time steps
init_tensor = torch.zeros(1000) #Grid size
i = 0
while i in range(init_tensor.shape[-1]-1):
    x = i / init_tensor.shape[-1]

    init_tensor[i] = np.sin(np.pi*x)
    i += 1


x = solve_pde(init_tensor, time)



