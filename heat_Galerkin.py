import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time
if __name__ == "__main__":
    import heatFenics as heatFenics
else:
    import heatFenics as heatFenics
'''
u(x,t) = (sum) yi(t) * Si(x)
w(x,t) = (sum) phij * Sj(x)
In weak sol: (integral) Si(x)Sj(x)dx * dyi/dt 
+ eta*(integral) dSi(x)/dx dSj(x)/dx dx * yi = 0  

'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
class ChebyshevInterpolation:
    def __init__(self, grid, sgrid):  
        
        self.grid = grid #Grid of the whole problem
        self.sgrid = sgrid #Grid of a singular shape function
        self.coeffNumber = self.grid.size(1)
        self.intPoints = self.sgrid.size(1)
        self.time_steps = self.grid.size(2)
        self.eta = 0.05
        self.dt = 1/self.time_steps
        
        self.epsilon = 10**(-6) 
        self.nsgrid = (2. * self.sgrid - 1.)/(1+self.epsilon) #Grid from -1 to 1
        self.coeffNumber = self.grid.size(1) #Nele 
         
        
        u0 = self.u_0_init_cond().to(device) #Apply Initial Condition to TrialFunction aka set y to y at time 0
        fenics_001 = torch.flipud(torch.load("heat_fenics_sol_tensor_0.01").to(device))
        fenics_01 = torch.flipud(torch.load("heat_fenics_sol_tensor_0.1").to(device))
        fenics_1 = torch.flipud(torch.load("heat_fenics_sol_tensor_1").to(device))
        fenics = torch.cat([fenics_001, fenics_01, fenics_1], dim=0)
        fenics = fenics.type(torch.float32)
        
        cheb_sol = fenics
        cheb_sol_mean = torch.mean(cheb_sol, axis=0)
        cheb_sol = cheb_sol - cheb_sol_mean

        U, S, V = torch.linalg.svd(cheb_sol)

        start = time.time()


        
        
        chebpoly = 3
        if chebpoly == 1: #Using standard Galerkin
            self.shapeFuncUnconstraint = self.ChebFunc().to(device) #Creating Shape function with Cheb Poly
            self.shapeFuncUnconstraintdx = self.ChebFuncdx().to(device) #Creating Derivative of Shape Func with Cheb Poly
            self.shapeFunc = self.shapeFuncUnconstraint * (-1)*(-1+(torch.pow(2*self.sgrid-1,2))) #Enforcing S(x) = 0 at 0 and 1
            self.shapeFuncdx = self.shapeFuncUnconstraintdx * (-1)*(-1+(torch.pow(2*self.sgrid-1,2))) + self.shapeFuncUnconstraint * (-1)*((8)*(self.sgrid)-4) #Enforcing d/dx S(x) = 0 at 0 and 1
        else:    #Using POD-Galerkin
            r = 1
            self.coeffNumber = r
            self.shapeFunc = V[:r, :]
            self.shapeFuncdx = self.central_difference_mat(V)[:r, :]

        
        self.M = self.MatrixM() #Init Matrix M
        self.C = self.MatrixC() #Init Matrix C
        
        self.y0 = self.y_0_init_cond().to(device)
        self.y_tot = self.EndResultVector().to(device)
        self.y_tot[0, :] = self.y0
        self.Minv = torch.inverse(self.M)
        for n in range(1, self.time_steps-1):
            self.y_tot[n, :] = self.Solve_t(self.y_tot[n-1])
        self.y_tot = torch.einsum('ij,jk->ik',self.y_tot,self.shapeFunc)
    
        end = time.time()
        plot = 1
        if plot == 1:   
            #Plotting the results
            fenics = heatFenics.solve_pde(u0, self.time_steps - 1).to(device)
            fenics = torch.flipud(fenics)
            fenics = fenics.type(torch.float32)
            print(f"Relative error : {(torch.mean(torch.abs(fenics-self.y_tot))/torch.mean(torch.abs(fenics)))*100}%") #Absolute Mean Difference between Fenics and Chebyshev
            print(f"Absolute mean error: {torch.mean(torch.abs(fenics-self.y_tot))}")
            plt.figure(figsize=(10,5), layout="tight")
            
            plt.subplot(1,2,2)
            plt.imshow(torch.abs(fenics-self.y_tot).cpu(), extent=[0, 1, 0, 1],cmap='jet', origin="lower")
            plt.colorbar(label='u(x,t)',pad=0.1, fraction=0.046)
            plt.title("Absolute Difference FEniCS-Chebyshev", y=1.05)
            
            min_value = torch.min(torch.min(self.y_tot),torch.min(fenics))
            max_value = torch.max(torch.max(self.y_tot),torch.max(fenics))
            
            plt.subplot(1,2,1)
            plt.imshow(self.y_tot.cpu(),cmap='jet' ,extent=[0, 1, 0, 1], vmin=min_value,vmax=max_value, origin="lower")
            plt.colorbar(label='u(x, t)',pad=0.1, fraction=0.046)
            plt.xlabel('x')
            plt.ylabel('t')
            plt.title('1D Burger with Chebyshev', y=1.05)
            
            plt.subplots_adjust(wspace=0.5)
            filename = 'heatgalerkincomparison'
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            
            plt.imshow(self.y_tot.cpu(), cmap='jet', extent=[0, 1, 0, 1], origin="lower")
            plt.colorbar(label='u(x, t)')
            plt.xlabel('x')
            plt.ylabel('t')
            filename='NumSol'
            plt.savefig(filename)
            
            plt.close()

            print("Computing time (SVD and timesteps):", end-start)
            
    
    

    def ChebFunc(self): #Chebyshev Polynomial of the first kind
        Sxn = torch.zeros(self.coeffNumber, self.intPoints)
        for n in range(0, self.coeffNumber):
            #Sxn[n, :] = 0.5*(1-(self.sgrid[0,:])*n)
            Sxn[n,:] = torch.cos(n * torch.arccos(self.nsgrid[0,:])) 
            
        return Sxn

    
    def ChebFuncdx(self): #Derivative of the Chebyshev Polynomial of the first kind
        Sdxn = torch.zeros(self.coeffNumber, self.intPoints)
        for n in range(0, self.coeffNumber):
            Sdxn[n,:] =  2 *  n *  torch.divide(torch.sin(n * torch.acos(self.nsgrid[0, :])),
                                        torch.sqrt(1 - torch.pow(self.nsgrid[0, :], 2))) 
        return Sdxn


    def EndResultVector(self): #Vector to store the results of the time steps
        y_tot = torch.zeros(self.time_steps-1, self.coeffNumber)
        return y_tot

    def y_0_init_cond(self): #Initial condition in the "coefficient space"
        u0 = torch.zeros(self.intPoints).to(device)
        for n in range(0, self.intPoints):
            u0[n] = torch.sin(torch.pi * self.sgrid[0, n])
        self.u0 = u0
        A = torch.einsum('ij,jk -> ik', self.shapeFunc, torch.transpose(self.shapeFunc,0,1)) #S*S^T
        y0 = torch.einsum('i,ij -> j',u0, torch.einsum('ij,jk -> ik',torch.transpose(self.shapeFunc,0,1),torch.inverse(A))) #Getting the coefficients from the Initial Condition
        return y0
    
    def u_0_init_cond(self): #Initial condition in the "real space"
        u0 = torch.zeros(self.intPoints).to(device)
        for n in range(0, self.intPoints):
            u0[n] = torch.sin(torch.pi * self.sgrid[0, n])
        self.u0 = u0
        return u0


    def central_difference_mat(self, A): #Central scheme for a matrix A
        h = 1/A.shape[1] #Matrix of shape [..., dx]
        A_diff = torch.zeros_like(A)
        A_diff[:, 1:-1] = (A[:, 2:] - A[:, :-2])/(2*h)
        return A_diff

 
    def MatrixM(self): #Setting up my matrix M        
        M = torch.einsum('ik,jk -> ijk', self.shapeFunc, self.shapeFunc)
        int_M = torch.trapz(M, self.sgrid)
        return int_M

    def MatrixC(self): #Setting up my matrix C
        C = torch.einsum('ik, jk -> ijk', self.shapeFuncdx, self.shapeFuncdx) 
        int_C = torch.trapz(C, self.sgrid)
        return int_C
    



    def Solve_t(self, y): #Solves for one time step
        
        expl = 0
        if expl == 1: #Explicit euler
            c = torch.einsum("ij, j -> i", self.C, y) * self.eta
            yt = (torch.einsum('ij, j -> i', self.Minv, -c))* self.dt + y #Full equation
            
        else: #Implicit Euler
            c = torch.einsum("ij,jk -> ik", self.Minv, self.C)
            impl_term = torch.inverse(torch.eye(c.shape[0]).to(device) + self.eta*self.dt*c)
            yt = torch.einsum("ij ,j -> i", impl_term, y)
        return yt  
    

nele = 14 #Number of elements -1 in one direction (so grid will be nele+1 * time_steps + 1)
time_steps = 1000
intPoints = 1000 #Number of elements in a single RBF (so one RBF will be intPoints * intPoints)
grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, nele + 1), torch.linspace(0, 1, time_steps + 1),
                                                  indexing='ij')

grid = torch.stack((grid_x,grid_y), dim=0).to(device)
sgrid_ = torch.meshgrid(torch.linspace(0, 1, intPoints))
sgrid = torch.stack(sgrid_, dim=0).to(device)

if __name__ == '__main__':
    ChebyshevInterpolation(grid,sgrid)
