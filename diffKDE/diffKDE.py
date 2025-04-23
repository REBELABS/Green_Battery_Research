import numpy as np
import math
from matplotlib import pyplot as plt
plt.matplotlib.rc('text', usetex=True)
from scipy import stats
from scipy import sparse
from scipy.integrate import trapz
from scipy.sparse.linalg import spsolve


def KDE(data, xmin = np.nan, xmax = np.nan, n = 1004, timesteps = 20, T=np.nan):

    opt_u, Omega, delta, p, stages, times = KDE_calculation(data, xmin, xmax, n, timesteps, T) #diffKDE calculation

    return opt_u, Omega




def KDE_calculation(data, xmin, xmax, n, timesteps, T):
    # the number of spatial discretization points will be n+1 inlcuding both boundary values and spliting the interval Omega in n equally sized subintervals of length h= (xmax-xmin)/n, 4 of them will be added outside [xmin, xmax]

    data, N, xmin, xmax = data_preparation(data, xmin, xmax) # reshape data
    Omega, h, n = grid(data, xmin, xmax, n) # set up spatial grid
    delta = initial_value(data, N, Omega, h, n) # calculate initial value
    p, f = pilot_estimation(data, N, delta, h, n, timesteps) # calculate pilot estimation steps
    A = matrix_A(h, n, p) # matrix for spatially discretized diffusion equation
    if (np.isnan(T)==True):
        T = bandwidth_selection(N, Omega, n, h, f, p) # approximation of optimal final iteration time
 
    ##### TIME STEPPING PRE-SET #####
    # initial value
    u_k = delta
    stages = [[u_k]]
    # initial time
    t = 0
    times = [[t]]
    #timesteps = 20 # time steps
    DELTA = T / timesteps # temporal increment
    
    ##### TIME FORWARD #####
    # time evolution with impl Euler
    B = sparse.identity(n+1) - sparse.csr_matrix(A).multiply(DELTA) # matrix for linear system for impl Euler
    while (t<T): # until final iteration time is rached
        u_k = spsolve(B, u_k) # solve by impl Euler
        t+=DELTA # update time
        stages.append([u_k])
        times.append([t])
    print('integral of the diffKDE =', trapz(u_k, x = Omega, dx=h)) # integral of diffusion KDE
    # append further solutions for family generation
    opt_u = u_k # save solution at optimal bandwidth
    T_custom = 2 * T # calculate to double time than optimum
    # next two lines to alternatively use also double timestep
    # DELTA_custom = T_custom /timesteps # double stepsize for second half, bc little changes there
    # B = sparse.identity(n + 1) - sparse.csr_matrix(A).multiply(DELTA_custom)  # matrix for linear system for impl Euler
    while (t<T_custom): # until final iteration time is reached
        u_k = spsolve(B, u_k) # solve by impl Euler
        t += DELTA
        # next line to alternatively use also double timestep
        # t += DELTA_custom # update time by same increment
        stages.append([u_k]) # save all individual approximation stages
        times.append([t]) # save times of the approximation stages

    
    return opt_u, Omega, delta, p, stages, times # diffKDE, grid, initial value, pilot, approximation stages, approximation timesteps




##### DATA PREPARATION ######
def data_preparation(data, xmin, xmax):
    # cast data to numpy array, if possible
    assert(isinstance(data, (np.ndarray, list)))
    if (type(data)==list):
        data = np.reshape(data, len(data))
    # restrict data to given range, if one is given
    if (np.isnan(xmin) == False):
        lower_mask = np.ma.masked_greater_equal(data, xmin)
        data = np.array(data)[lower_mask.mask]
    if (np.isnan(xmax) == False):
        upper_mask = np.ma.masked_less_equal(data, xmax)
        data = np.array(data)[upper_mask.mask]
    # if no range is handed in, the default interval is set to the data range
    if np.isnan(xmin):
        xmin = min(data)
    if np.isnan(xmax):
        xmax = max(data)
    # assure that interval is non-empty
    assert (xmin <= xmax)
    N = len(data)
    assert(N != 0)
    print('number of used data points: N =', N)
    
    return data, N, xmin, xmax


##### SPATIAL DISCRETIZATION ######
def grid(data, xmin, xmax, n):
    h = (xmax-xmin)/(n-4) # inner spatial discretization fineness; we use four less to let them later be included outside the data range to ensure no data point is a boundary value
    Omega = np.arange(xmin, xmax, h) # inner spatial discretization
    Omega = np.concatenate((Omega, [xmax])) # include rightmost point
    Omega = np.concatenate(([xmin - h - h], [xmin - h], Omega, [xmax + h], [xmax + h + h])) # add four additional discr points outide [xmin, xmax]
    print('number spatial discretization points: n+1 =', n+1)
    print('spatial discretization step size: h =', h)
    print('spatial discretization area: Omega =',Omega)

    return Omega, h, n
    

##### INITIAL VALUE CALCULATION #####
def initial_value(data, N, Omega, h, n):
    delta = np.zeros(n+1) # initialize with zero
    for j in range(N): # iterate over data points X
            i = np.searchsorted(Omega, data[j]) # find position i of X_j in the saptial discr Omega
            delta[i-1]+= 1/(h**2) * (Omega[i] - data[j]) # add at left neighbour from X_j value H_i-1
            delta[i]+= 1/(h**2) * (data[j] - Omega[i-1]) # add at right neighbour from X_j value H_i
    delta/=N # norm to get integral=1
    print('integral of the initial value =', trapz(delta, x = Omega, dx=h)) # integral of initial value
        
    return delta
    
    
##### PILOT ESTIMATES CALCULATION #####
# solve the diffusion equation du/dt=0.5*d^2u/dx^2 to get pilot estimates for p and f:
def pilot_estimation(data, N, delta, h, n, timesteps):
    # set up matrix A for calculation of pilot steps with parameter function = 1
    q=np.ones(n+1)
    Apilot = matrix_A(h, n, q) # matrix for spatially discretized diffusion equation
    # time stepping pre-setting
    t = 0 # initial time
    #timesteps = 20 # number of
    # setup for pilot p
    p = delta # initial value
    T_pilot = np.std(data)**2 * ((4/3)*N)**(-2/5) # bandwidth (Silverman, 1986)
    print('T_p=', T_pilot)
    DELTA = T_pilot / timesteps # temporal increment for pilot p
    Bpilot = sparse.identity(n+1) - sparse.csr_matrix(Apilot).multiply(DELTA) # matrix for linear system for impl Euler for pilot p
    # setup for pilot f
    f = delta # initial value
    T_f = (0.9*min(np.std(data), stats.iqr(data)/1.34))**2 * N**(-2/5) # bandwidth "Silverman2" (Sheather, 2004)
    print('T_f=',T_f)
    DELTA_f = T_f / timesteps # temporal increment for pilot f
    Bf = sparse.identity(n+1) - sparse.csr_matrix(Apilot).multiply(DELTA_f) # matrix for linear system for impl Euler for pilot f
    # solve system with impl Euler for both pilot estimates
    while (t<T_pilot): # until final iteration time is rached
        p = spsolve(Bpilot, p) # solve for p by impl Euler
        f = spsolve(Bf, f) # solve for f by impl Euler
        t+=DELTA # update time
    # catch p=0 for matrix calculation below
    if (min(p)==0):
        p = np.ones(n+1) # use p=1
        print('min(p)=0.0 setting p:=1')
        
    return p, f
    
    
    
####### matrix A for diffKDE calculation #####
def matrix_A(h, n, p):
    diag_1 = np.concatenate(([2 / p[1]], 1 / p[2:])) # upper side diagonal
    diag_2 = - 2 / p # main diagonal
    diag_3 = np.concatenate(( 1 / p[:n-1], [2 / p[n-1]])) # lower side diagonal
    diag_rows = np.array([diag_1,diag_2,diag_3], dtype=object) # create vector with all three diags
    positions = [1, 0, -1] # create vector to assign positions of the three diags
    A = sparse.diags(diag_rows, positions, shape=(n+1, n+1), format = 'csr') # set up matrix A as sparse
    A = sparse.csr_matrix(A).multiply(0.5 * (1/h**2)) # include multilplicative factor leaving A sparse
    
    return A
    
    
##### BANDWIDTH SELECTION FOR diffKDE #####
def bandwidth_selection(N, Omega, n, h, f, p):
    # approximate (f/p)" by finite differences
    quot_curv = []
    # used homogeneous Neumann boundary conditions for boundary values
    quot_curv.append((2 * f[1] / p[1] - 2 * f[0] / p[0]) / (h**2))
    for i in range(1,n): # grenzen bitte ueberpruefen 0-n-2 = 1002
        quot_curv.append((f[i+1] / p[i+1] - 2 * f[i] / p[i] + f[i-1] / p[i-1]) / (h**2))
    quot_curv.append((2 * f[n-1] / p[n-1] - 2 * f[n] / p[n]) / (h**2))
    quot_curv = np.multiply(quot_curv, quot_curv) # square of (f/p)"
    quot_curv = trapz(quot_curv, x = Omega, dx=h) # integral of ((f/p)")^2 over Omega with discretization h
    quot_curv = math.sqrt(quot_curv) # root of integral
    E_sigma = 0 # initialization of nominator
    for i in range(0,n+1): # evaluation at every grid point
        E_sigma += (1/(n+1)) * math.sqrt(p[i])
    T = (E_sigma / (2 * N * math.sqrt(math.pi) * quot_curv**2))**(2/5) # bandwidth for diffKDE
    if (np.isnan(T)): # catch NaN
        T = T_f # use bandwidth from above
        print('T*=nan! setting T*:=T_f')
    print('T*=', T)
    return T
    
    
    
##### PLOT FUNCTION FOR PILOT p AND diffKDE ####
def pilot_plot(data, xmin = np.nan, xmax = np.nan, n = 1004, timesteps = 20, T=np.nan):

    opt_u, Omega, delta, p, stages, times = KDE_calculation(data, xmin, xmax, n, timesteps, T) #diffKDE calculation
    # figure set-up
    cm = plt.get_cmap('viridis')
    fig, ax = plt.subplots()
    ax.plot(data, np.zeros(len(data)), 'ko', alpha=0.2, label = 'data points') # plot data points
    plt.plot(Omega, p, color='#d62728', label='pilot') # plot pilot
    plt.plot(Omega, opt_u, color='#1f77b4', label='diffKDE')
    plt.legend()
    plt.grid()
    plt.xlabel(r'data value')
    plt.ylabel('density')
    plt.title('The diffusion KDE and its pilot estimate')
    plt.show()
    
    return
    


##### PLOT FUNCTION FOR EVOLUTION SEQUENCE ####
def evol_plot(data, xmin = np.nan, xmax = np.nan, n = 1004, timesteps = 20, T=np.nan):

    opt_u, Omega, delta, p, stages, times = KDE_calculation(data, xmin, xmax, n, timesteps, T) #diffKDE calculation
    #figure set-up
    cm = plt.get_cmap('viridis')
    fig, ax = plt.subplots()
    ax.plot(data, np.zeros(len(data)), 'ko', alpha=0.2, label = 'data points') # plot data points
    ax.plot(Omega, delta, alpha =0.8, linewidth =0.5, color=cm(0.9), label='initial values') # plot initial value
    for i in range(1, timesteps):
            ax.plot(Omega, stages[i][0], alpha =0.8, color=cm(1-i/timesteps))# plot each evolution step
    
    plt.plot(Omega, opt_u, color='#1f77b4', label='diffKDE') # plot diffKDE
    plt.ylim([-0.03*np.max(opt_u), np.max(opt_u) + 0.2*np.max(opt_u)])
    plt.legend()
    plt.grid()
    plt.xlabel(r'data value')
    plt.ylabel('density')
    plt.title('The diffusion KDE and its evolution stages')
    plt.show()
    
    return
    
    
    
def custom_plot(data, xmin = np.nan, xmax = np.nan, n = 1004, timesteps = 20, T=np.nan):
    from matplotlib.widgets import Slider, Button
        
    opt_u, Omega, delta, p, stages, times = KDE_calculation(data, xmin, xmax, n, timesteps, T) #diffKDE calculation
    
    times = np.array(times).flatten() # get times
    def f(t): # interactive choice of diffKDE at different times
        t = np.absolute(times-t).argmin() # get index of current time in times&stages vector
        print('timestep=', t)
        return stages[t][0]
    init_t = max(times)*0.5 # initial time=final time of standard calculation ie at the opt BW
    fig, ax = plt.subplots()
    line, = plt.plot(Omega, f(init_t)) # initial plot
    ax.set_xlabel('data values')
    ax.set_ylabel('density')
    plt.subplots_adjust(bottom=0.25)
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    # slider to select the evolution time
    t_slider = Slider(
        ax=axfreq,
        label='time [T]',
        valmin=min(times),
        valmax=max(times),
        valfmt='%.5f', # shows BW displayed in GUI as floating point with 5 digits
        valinit=init_t,
    )
    # update function for slider values
    def update(val):
        line.set_ydata(f(t_slider.val))
        fig.canvas.draw_idle()
    t_slider.on_changed(update) # link slider to update function
    # reset button to initial time/diffKDE
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, r'$T^*$', hovercolor='0.975')
    def reset(event):
        t_slider.reset()
    button.on_clicked(reset)
    fig.suptitle('The diffusion KDE at different times')
    plt.show()

    return
    
