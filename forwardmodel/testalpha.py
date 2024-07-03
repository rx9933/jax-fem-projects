import numpy as onp
import jax.numpy as np
def get_alpha(x0):
    vi = onp.loadtxt('cell_vertices_initial.txt') 
    rff = 100 # characteristic distance "ff" for farfield
    vi = np.array(vi) # mesh vertices on cell surface
    rs = np.min(np.linalg.norm(x0-vi)) # distance to cell surface
    # print("rs",onp.array(rs))
    # print("rff",onp.array(rff))
    rsc = np.minimum(rs, rff) # clipped distance to cell surface
    a0 = 2.5
    rcrit = rff*np.sqrt(2*a0-1)/(np.sqrt(2*a0-1)+1) #characterestic distance for most degraded gel portion
    aideal = 1/2*(((rsc-rcrit)/(rff-rcrit))**2 + 1)

    return aideal