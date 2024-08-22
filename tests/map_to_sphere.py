from gconv import icosahedron
import numpy as np
from stripy import sTriangulation
from stripy.spherical import xyz2lonlat,lonlat2xyz
import matplotlib.pyplot as plt
from gconv.dihedral12 import xy2ind
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator



def makeBvecMeshes(x,y,z):
    plons, plats = xyz2lonlat(x,y,z)
    nlons, nlats = xyz2lonlat(-x,-y,-z)
    lons = np.concatenate((plons,nlons),0)
    lats = np.concatenate((plats, nlats), 0)
    return sTriangulation(lons,lats,tree=True)

def makeInverseDistInterpMatrix(bvec_mesh,ico):
    #source target
    N_ico=len(ico.X_in_grid.flatten())
    N_bvec=len(bvec_mesh.lons)
    print('shape is: %d, %d'%(N_ico,N_bvec))
    #initiate inpterp matrix
    interp_matrix = np.zeros([N_ico, N_bvec])
    #get nearest vertices and distances
    lons_,lats_=xyz2lonlat(ico.X_in_grid.flatten(),ico.Y_in_grid.flatten(),ico.Z_in_grid.flatten())
    dist,idx=bvec_mesh.nearest_vertices(lons_,lats_,k=3)
    # dist,idx=bvec_mesh.nearest_vertices(ico_mesh.lons,ico_mesh.lats,k=3)
    weights=1/(dist)
    for row in range(0,N_ico):
        norm=sum(weights[row])
        interp_matrix[row,idx[row]]=weights[row]/norm
    return interp_matrix

def sphere_to_flat_basis(ico_mesh):
    H = ico_mesh.m + 1
    w = 5 * (H + 1)
    h = H + 1
    basis = np.empty([h, w])
    basis[:]=np.nan
    top_faces = [[1, 2], [5, 6], [9, 10], [13, 14], [17, 18]]
    for c in range(0, 5):
        face = top_faces[c]
        for top_bottom in face:
            #signal_inds are the inds that are needed from vector to matrix
            signal_inds = np.asarray( ico_mesh.interpolation_inds[top_bottom]).astype(int)
            #signal = ico_signal[signal_inds]
            i = ico_mesh.i_list[top_bottom]
            j = ico_mesh.j_list[top_bottom]
            i = np.asarray(i).astype(int)
            j = np.asarray(c * h + j + 1).astype(int)
            basis[i, j] = signal_inds
    #for padding
    strip_xy = np.arange(0, H - 1)
    for c in range(0, 5):  # for padding
        basis[0, c * h + 1] = 0  # northpole
        c_left = c
        x_left = -1
        y_left = strip_xy
        i_left, j_left = xy2ind(H, c_left, x_left, y_left)
        c_right = (c - 1) % 5
        x_right = H - 2 - strip_xy
        y_right = H - 2
        i_right, j_right = xy2ind(H, c_right, x_right, y_right)
        basis[i_left, j_left] = basis[i_right, j_right]
    return basis


def gaussian_2d(x, y, mu_x=np.pi, mu_y=0, sigma_x=.3, sigma_y=.3):
    # Compute the Gaussian function
    g1 = np.exp(-(((x - mu_x)**2) / (2 * sigma_x**2) + ((y - mu_y)**2) / (2 * sigma_y**2)))
    mu_x = mu_x + np.pi
    mu_y = np.pi - mu_x
    g2 = np.exp(-(((x - mu_x)**2) / (2 * sigma_x**2) + ((y - mu_y)**2) / (2 * sigma_y**2)))
    return (g1+g2)/2

def antipodal_gaussian(theta, phi, theta0=0, phi0=0, sigma=0.1):
    """
    Returns the value of an antipodally identified Gaussian on the sphere.

    Parameters:
    - theta: Polar angle (colatitude) of the point in radians.
    - phi: Azimuthal angle (longitude) of the point in radians.
    - theta0: Polar angle (colatitude) of the Gaussian peak in radians.
    - phi0: Azimuthal angle (longitude) of the Gaussian peak in radians.
    - sigma: Width (standard deviation) of the Gaussian.

    Returns:
    - The value of the antipodal Gaussian at the specified (theta, phi).
    """
    # Compute the Euclidean distance on the unit sphere between (theta, phi) and (theta0, phi0)
    cos_gamma = np.sin(theta) * np.sin(theta0) * np.cos(phi - phi0) + np.cos(theta) * np.cos(theta0)
    gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
    
    # Compute the Euclidean distance on the unit sphere between (theta, phi) and the antipodal point of (theta0, phi0)
    gamma_antipodal = np.pi - gamma
    
    # Gaussian on the sphere: g(theta, phi) + g(pi - theta, phi + pi)
    gaussian_value = np.exp(-gamma**2 / (2 * sigma**2)) + np.exp(-gamma_antipodal**2 / (2 * sigma**2))
    
    return gaussian_value

#get the icomeshes and bvec meshes
ico_mesh=icosahedron.icomesh(20)
N=50
lons=np.linspace(-np.pi, np.pi,N)
lats=np.linspace(-np.pi/2,np.pi/2,N)
lons,lats=np.meshgrid(lons,lats)



out=antipodal_gaussian(lons.flatten(),lats.flatten()).reshape(lons.shape)

plt.figure()
plt.imshow(out)
plt.savefig('on_the_sphere.png')


# x,y,z=lonlat2xyz(lons.flatten(),lats.flatten())


# bvec_mesh=makeBvecMeshes(x,y,z)
# # interp_matrix=makeInverseDistInterpMatrix(bvec_mesh,ico_mesh)

# # #hold onto signal for later
# signalp=gaussian_2d(lons,lats)
# lons_,lats_=xyz2lonlat(-x,-y,-z)
# signalp1=gaussian_2d(lons_,lats_)

# signal_cat=np.concatenate((signalp.flatten(),signalp1.flatten()))


# xyz=np.concatenate([[x,y,z],[-x,-y,-z]],1)


# interpolator = NearestNDInterpolator(xyz.T, signal_cat)

# #signal_cat=100*np.zeros_like(signalp)
# # lons_,lats_=xyz2lonlat(ico_mesh.X_in_grid.flatten(),ico_mesh.Y_in_grid.flatten(),ico_mesh.Z_in_grid.flatten())


# # out_flat=bvec_mesh.interpolate_linear(lons_,lats_,signal_cat)[0].reshape(ico_mesh.X_in_grid.shape)

# out_flat=interpolator(ico_mesh.X_in_grid.flatten(),ico_mesh.Y_in_grid.flatten(),ico_mesh.Z_in_grid.flatten()).reshape(ico_mesh.X_in_grid.shape)


# plt.figure()
# plt.imshow(out_flat)
# plt.savefig('on_the_plane.png')



# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(signalp)
# plt.savefig('on_the_sphere.png')

# basis=sphere_to_flat_basis(ico_mesh)

# #project the signal
# out=interp_matrix @ signal_cat.flatten()

# # out= 0.5*(out + out[ico_mesh.antipodals])


# # h=basis.shape[-2]
# # w=basis.shape[-1]
# # out_flat=np.zeros([h,w])

# # i_nan,j_nan=np.where(np.isnan(basis))
# # basis[i_nan,j_nan]=0
# # basis=basis.astype(int)

# # out_flat=out[basis]

# # 
# out_flat=np.zeros_like(ico_mesh.X_in_grid)

lons_,lats_=xyz2lonlat(ico_mesh.X_in_grid.flatten(),ico_mesh.Y_in_grid.flatten(),ico_mesh.Z_in_grid.flatten())

signalp=antipodal_gaussian(lons_,lats_)

out_flat=signalp.reshape(ico_mesh.X_in_grid.shape)



plt.figure()
plt.imshow(out_flat)
plt.savefig('on_the_plane1.png')
