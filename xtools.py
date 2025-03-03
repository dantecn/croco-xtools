##########################################
# CROCO TOOLS with xarray
##########################################
import xarray as xr
import numpy as np
import xoak
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import os

###########################################################
# Functions to adapt croco outputs ########################
###########################################################
def adjust_coords(ds):

    if ('nav_lon_u' not in ds.coords) & ('nav_lon_rho' not in ds.coords): # nav_lon_rho? #nav_lon_u
        ##########################
        print('for regular CROCO files')
        ds = ds.set_coords([c for c in ds.variables if 'lon' in c or 'lat' in c ])
    else:
        ##########################
        print('for XIOS files')
        
        if 'time' in ds.variables:
            ds = ds.drop('time')
            
        ds = ds.rename({'time_counter': 'time'})
        
        ds = ds.reset_coords([c for c in ds.coords if 'nav' in c])

        # rename redundant dimensions
        _dims = (d for d in ['x_v', 'y_u', 'x_w', 'y_w'] if d in ds.dims)
        for d in _dims:
            ds = ds.rename({d: d[0]+'_rho'})

        # change axis names to xi,eta (instead of x,y)
        _dims = (d for d in ['x_u', 'x_rho'] if d in ds.dims)
        for d in _dims:
            ds = ds.rename({d: 'xi' + d[1:]}) 

        _dims = (d for d in ['y_v', 'y_rho'] if d in ds.dims)
        for d in _dims:
            ds = ds.rename({d: 'eta' + d[1:]}) 


        # change nav variables to coordinates        
        _coords = [d for d in [d for d in ds.data_vars.keys()] if "nav_" in d]
        ds = ds.set_coords(_coords) 

        # rename coordinates 
        for c in ds.coords:
            new_c = c.replace('nav_lat','lat').replace('nav_lon','lon')
            ds = ds.rename({c:new_c})
            # reset names and units
            ds[new_c] = (ds[new_c].assign_attrs(units='deg', 
                                               standard_name=new_c,
                                               long_name=new_c)
                        )

    ##########################
    # For all types  
    if 'eta_psi' in ds.dims: ds = ds.rename({'eta_psi': 'eta_v'}) 
    if 'xi_psi' in ds.dims: ds = ds.rename({'xi_psi': 'xi_u'}) 
    
    ##########################
    # Make all variables float32
#     for var in list(ds.variables):
#         if ds[var].dtype=='float64':
#             ds[var] = ds[var].astype('float32')
    
    '''    ##########################
    # Create xgcm grid
    coords={'xi':{'center':'xi_rho', 'inner':'xi_u'}, 
            'eta':{'center':'eta_rho', 'inner':'eta_v'}, 
            's':{'center':'s_rho', 'outer':'s_w'}}

    ds.attrs['xgcm-Grid'] = Grid(ds, coords=coords, periodic=[])
    '''

    return ds

########
def add_grd(ds,grd):
    
    ##########################
    for variable in grd.data_vars.keys():
        #print(variable)
        ds[variable] = grd[variable]
        
    ds['mask_rho'] = ds.mask_rho.where(ds.mask_rho>0,np.nan)
    
    if 'lon_psi' not in ds.coords: 
        #ds['lon_psi'] = grd['lon_psi']
        #ds['lat_psi'] = grd['lat_psi']
        ds = ds.assign_coords({'lon_psi':grd['lon_psi'], 'lat_psi':grd['lat_psi']})

    ##########################
    # Make all variables float32 (again)
#     for var in list(ds.variables):
#         if ds[var].dtype=='float64':
#             ds[var] = ds[var].astype('float32')
        
    return ds
 
########
def calc_vertical_coord(ds):
    
    if 'CPP-options' in ds.attrs:
        cpp = 'CPP-options'
    else:
        cpp = 'CPPS'
    
    if 'VertCoordType' in ds.attrs:
        if ds.VertCoordType=='NEW':
            ds['Vtransform'] = 2
        else:
            ds['Vtransform'] = 1
    elif 'NEW_S_COORD' in ds.attrs[cpp]:
        ds['Vtransform'] = 2
    else:
        ds['Vtransform'] = 1

    #####################

    if ('sc_r' not in ds):
        # need to define sc_r for older roms files
        N = ds.s_rho.shape[0]
        ds['sc_r'] = xr.DataArray((np.arange(N) - N +0.5) / N,  dims=["s_rho"])
        ds['sc_w'] = xr.DataArray((np.arange(N+1) - N) / N,  dims=["s_w"])
    else: # from grd
        ds['sc_r'] = xr.DataArray(ds.sc_r,  dims=["s_rho"])
        ds['sc_w'] = xr.DataArray(ds.sc_w,  dims=["s_w"])          
#         ds['Cs_r'] = xr.DataArray(ds.Cs_r,  dims=["s_rho"])
#         ds['Cs_w'] = xr.DataArray(ds.Cs_w,  dims=["s_w"]) 

    ds['Cs_r'] = xr.DataArray(ds.Cs_r,  dims=["s_rho"])
    ds['Cs_w'] = xr.DataArray(ds.Cs_w,  dims=["s_w"])
    
    #####################
    # making all variables float32
#     tof32 = lambda da: da.astype('float32')
#     [ds['sc_r'],ds['sc_w'],ds['h'],
#      ds['hc'],ds['Cs_r'],ds['Cs_w']] = map(tof32,[ds.sc_r,ds.sc_w,ds.h,ds.hc,ds.Cs_r,ds.Cs_w])

    '''try:
        ds = ds.reset_coords([c for c in ds.coords if 'Cs' in c])
    except:
        pass'''

    #####################
    # Including vertical coordinates

    if ds.Vtransform == 1:
        Zo_rho = ds.hc * (ds.sc_r - ds.Cs_r) + ds.Cs_r * ds.h
        z_rho = Zo_rho + ds.zeta * (1 + Zo_rho/ds.h)
    elif ds.Vtransform == 2:
        Zo_rho = (ds.hc * ds.sc_r + ds.Cs_r * ds.h) / (ds.hc + ds.h)
        z_rho = ds.zeta + (ds.zeta + ds.h) * Zo_rho; Zo_rho=None

    # ds.coords['z_rho'] = z_rholon
    
    if ds.Vtransform == 1:
        Zo_w = ds.hc * (ds.sc_w - ds.Cs_w) + ds.Cs_w * ds.h
        z_w = Zo_rho + ds.zeta * (1 + Zo_w/ds.h)
    elif ds.Vtransform == 2:
        Zo_w = (ds.hc * ds.sc_w + ds.Cs_w * ds.h) / (ds.hc + ds.h)
        z_w = ds.zeta + (ds.zeta + ds.h) * Zo_w; Zo_w=None

    # ds.coords['z_w'] = z_w
    
    return z_rho, z_w

########
def add_vertical_coord(ds,which='both'):

    if which=='both': ds.coords['z_rho'], ds.coords['z_w'] = calc_vertical_coord(ds)
    elif which=='z_rho': ds.coords['z_rho'], _ = calc_vertical_coord(ds)
    elif which=='z_w': _,ds.coords['z_w'] = calc_vertical_coord(ds)
        
    return ds

########
def get_vertical_section(var,lon_obs,lat_obs,coords=["lat_rho","lon_rho"],newdim="no",dist=True):

    # add "minimum distance" to xoak
    ireg = xoak.IndexRegistry()
    @ireg.register('brute_force')
    class BruteForceIndex(xoak.IndexAdapter):
        """Brute-force nearest neighbor lookup."""
        def build(self, points):
            # there is no index to build here, just return the points
            return points
        def query(self, index, points):
            positions, distances = pairwise_distances_argmin_min(points, index)
            return distances, positions
    
    # rename dimensions
    var = var.rename({coords[0]:'cy',coords[1]:'cx'})
    
    # make lon_obs and lat_obs xrarray with dims 'newdim'
    lon_obs=xr.DataArray(lon_obs,dims=newdim) if type(lon_obs) is not xr.core.dataarray.DataArray else lon_obs
    lat_obs=xr.DataArray(lat_obs,dims=newdim) if type(lat_obs) is not xr.core.dataarray.DataArray else lat_obs
    
    # add brute force indexes
    var.xoak.set_index(["cy", "cx"],ireg.brute_force)
    
    # get section
    var_sec = var.xoak.sel(cy=lat_obs,cx=lon_obs)
    
    # rename dimensions back
    var_sec = var_sec.rename({"cy":coords[0],"cx":coords[1]})
    
    # calculate section distance and add coordinates
    if dist: # xtools use calc_dist method
        
        deg2m_lon = 60*1852*np.cos(np.deg2rad(lat_obs + lat_obs.shift({newdim:1}))/2).dropna(newdim)
        deg2m_lat = 60*1852
    
        # distance
        dlon = (deg2m_lon*(lon_obs.data[1:]-lon_obs.data[:-1]))**2
        dlat = (deg2m_lat*(lat_obs.data[1:]-lat_obs.data[:-1]))**2
    
        distkm = 1e-3*xr.concat([xr.DataArray(0),np.cumsum(np.sqrt(dlon + dlat))],dim=newdim)
        
        # add to dataarray
        var_sec = var_sec.assign_coords(dist=distkm)
    
    return var_sec
        
#######################################################
# Compute density and Brunt-Vaissala frequency
#######################################################
def rho_eos(Tt,Ts,z_r,g,rho0,z_w=None,sig0=True):

    '''
    Compute rho from equation of state rho on rho (vert and hor) grid
    bvf is computed only if z_w is not None bvf computed on rho-w grid
    (first and last levels set to 0) 
    '''
    
#     if np.rank(Tt)==2:
#         [M,L]=Tt.shape
#     else:
#         [N,M,L]=Tt.shape

    A00=+19092.56;A01=+209.8925;
    A02=-3.041638;A03=-1.852732e-3;A04=-1.361629e-5;A10=104.4077;
    A11=-6.500517;A12=+0.1553190;A13=2.326469e-4;AS0=-5.587545;
    AS1=+0.7390729;AS2=-1.909078e-2;B00=+4.721788e-1;B01=+1.028859e-2;
    B02=-2.512549e-4;B03=-5.939910e-7;B10=-1.571896e-2;B11=-2.598241e-4;
    B12=+7.267926e-6;BS1=+2.042967e-3;E00=+1.045941e-5;E01=-5.782165e-10;
    E02=+1.296821e-7;E10=-2.595994e-7;E11=-1.248266e-9;E12=-3.508914e-9;

    QR=+999.842594;Q01=+6.793952e-2;Q02=-9.095290e-3;
    Q03=+1.001685e-4;Q04=-1.120083e-6;Q05=+6.536332e-9;Q10=+0.824493;
    Q11=-4.08990e-3;Q12=+7.64380e-5;Q13=-8.24670e-7;Q14=+5.38750e-9;
    QS0=-5.72466e-3;QS1=+1.02270e-4;QS2=-1.65460e-6;Q20=+4.8314e-4;

    sqrtTs=Ts ** 0.5;
    
    K0=A00+Tt*(A01+Tt*(A02+Tt*(A03+Tt*A04)))\
    +Ts*(A10+Tt*(A11+Tt*(A12+Tt*A13))\
    +sqrtTs*(AS0+Tt*(AS1+Tt*AS2)));
    
    K1=B00+Tt*(B01+Tt*(B02+Tt*B03))\
    +Ts*(B10+Tt*(B11+Tt*B12)+sqrtTs*BS1);
    
    K2=E00+Tt*(E01+Tt*E02)\
    +Ts*(E10+Tt*(E11+Tt*E12));
    
    rho1=QR+Tt*(Q01+Tt*(Q02+Tt*(Q03+Tt*(Q04+Tt*Q05))))\
    +Ts*(Q10+Tt*(Q11+Tt*(Q12+Tt*(Q13+Tt*Q14)))\
    +sqrtTs*(QS0+Tt*(QS1+Tt*QS2))+Ts*Q20);

    rho=rho1/(1+0.1*z_r/(K0-z_r*(K1-z_r*K2)));

    #######################################################

    if z_w!=None:
        
        [K0,K1,K2,z_r,rho1]=map(lambda a:a.rename({'s_rho':'z'}),[K0,K1,K2,z_r,rho1])
        z_w=z_w.rename({'s_w':'z'})

        # cut zw
        z_w = z_w.isel(z=slice(None,-1))
        # coef
        cff=g/rho0;
        
        n2m=-cff*(rho1.isel({'z':slice(1,None)})/\
        (1.+0.1*z_w.isel({'z':slice(1,None)})/\
        ( K0.isel({'z':slice(1,None)})-z_w.isel({'z':slice(1,None)})*(K1.isel({'z':slice(1,None)})-z_w.isel({'z':slice(1,None)})*K2.isel({'z':slice(1,None)}))))\
        -rho1.isel({'z':slice(None,-1)})/( 1.+0.1*z_w.isel({'z':slice(1,None)})/\
        ( K0.isel({'z':slice(None,-1)})-z_w.isel({'z':slice(1,None)})*(K1.isel({'z':slice(None,-1)})-z_w.isel({'z':slice(1,None)})*K2.isel({'z':slice(None,-1)})))))\
        /(z_r.isel({'z':slice(1,None)})-z_r.isel({'z':slice(None,-1)}));

        # add up and down boundary conditions
        bvf = xr.concat([xr.zeros_like(z_w.isel(z=0)),tst,xr.zeros_like(z_w.isel(z=0))],dim='z')
        bvf = bvf.rename({'z':'z_w'})
        
        if sig0:
            return [rho1,bvf]
        else:
            return [rho,bvf]

    else:
        if sig0:
            return rho1
        else:
            return rho

#######################################################
# Compute vertical velocity
#######################################################
def getw(u,v,pm,pn,z_r,z_w):

    # drop coordinates
    [u,v,z_r,z_w] = map(lambda a: a.drop([cc for cc in a.coords]),[u,v,z_r,z_w])

    # calc w
    flxu = ((rho2u(z_w.shift(s_w=-1) - z_w).drop_isel(s_w=-1).rename({'s_w':'s_rho'})/
            (0.5*(pn+pn.shift(xi_rho=-1)).drop_isel(xi_rho=-1).rename({'xi_rho':'xi_u'})))*
            u
           )
    flxv = ((rho2v(z_w.shift(s_w=-1) - z_w).drop_isel(s_w=-1).rename({'s_w':'s_rho'})/
            (0.5*(pm+pm.shift(eta_rho=-1)).drop_isel(eta_rho=-1).rename({'eta_rho':'eta_v'})))*
            v
            )
    
    wrk = -1*(
          (pm*pn).isel(eta_rho=slice(1,None),xi_rho=slice(1,None))*
          ((flxu.shift(xi_u=-1)-flxu).isel(eta_rho=slice(1,None)).rename({'xi_u':'xi_rho'})+
           (flxv.shift(eta_v=-1)-flxv).isel(xi_rho=slice(1,None)).rename({'eta_v':'eta_rho'}))
          )#.drop_isel(eta_rho=-1,xi_rho=-1)
    
    # add nan in rho grid points
    wrk = xr.concat([wrk.shift(eta_rho=-1).isel(eta_rho=-1), wrk],dim='eta_rho')
    wrk = xr.concat([wrk.shift(xi_rho=-1).isel(xi_rho=-1), wrk],dim='xi_rho')

    # integrate with zero at the bottom (s_w)
    wrk = xr.concat([xr.zeros_like(wrk.isel(s_rho=0)),wrk.cumsum('s_rho')],dim='s_rho')
  
    wrk = wrk.chunk({'eta_rho':wrk.eta_rho.size,'xi_rho':wrk.xi_rho.size})
    
    if wrk.s_rho.size==2:
        wvlc = wrk.copy()
    else:
        # move to vertical rho points

        # from s_rho=2 on
        wvlc = 0.5625*(wrk.shift(s_rho=1)+wrk)-0.0625*(wrk.shift(s_rho=2)+wrk.shift(s_rho=-1))
        # first and last cells of s_rho
        wvlc = wvlc.drop_isel(s_rho=[0,1,-1])
        wvlc = xr.concat([-0.125*wrk.isel(s_rho=2) + 0.75*wrk.isel(s_rho=1) + 0.375 * wrk.isel(s_rho=0),
                          wvlc,
                          0.375*wrk.isel(s_rho=-1) + 0.75*wrk.isel(s_rho=-2) - 0.125 * wrk.isel(s_rho=-3)],
                         dim='s_rho')
        
    #add contributions due to S-coord slopes (u*dz/dx and v*dz/dy)
    Wx =(xr.concat([u,u.shift(xi_u=-1).isel(xi_u=-1)],dim='xi_u').rename({'xi_u':'xi_rho'})*
        (z_r.shift(xi_rho=-1) - z_r)*
        (pm+pm.shift(xi_rho=-1))
        )
    Wy =(xr.concat([v,v.shift(eta_v=-1).isel(eta_v=-1)],dim='eta_v').rename({'eta_v':'eta_rho'})*
        (z_r.shift(eta_rho=-1) - z_r)*
        (pn+pn.shift(eta_rho=-1))
        )
    
    Wy,Wx = Wy.chunk({'eta_rho':Wy.eta_rho.size}),Wx.chunk({'xi_rho':Wx.eta_rho.size})
    
    wvlc = wvlc + 0.25 * (Wy.shift(eta_rho=1)+Wy+Wx.shift(xi_rho=1)+Wx)

    return wvlc


###########################################################
# Functions to slice croco outputs ########################
###########################################################
def interpolate_vertically(vals,zvals,znew,dimz='s_rho',ztype='depth',interp_boundaries=True):
    
    '''
    Interpolates vertically to z-levels (or rho) levels
    
    INPUT:
        vals:  property to interpolate
        zvals: depths (or denisty) in the same grid as vals
        znew = target depth(s) or densities. List, Array, or DataArray
               znew=xr.DataArray(np.array([depths/znew]),dims='s_rho') with vertical coord s_rho.
    
    OUTPUT:
        vals_i: vals at znew grid
    
    slightly adapted from René Schubert's function
    '''
    
    # make znew xrarray with dims 's_rho'
    znew=xr.DataArray(np.array(znew),dims=dimz) if type(znew) is not xr.core.dataarray.DataArray else znew
    
    # create dummy dimension dimz
    znew=znew.rename({dimz:'dimz'})
    zvals=zvals.rename({dimz:'dimz'})
    vals=vals.rename({dimz:'dimz'})
    
    # indez for all depths
    k = xr.DataArray(np.arange(0,znew.dimz.size,1),dims='s_rho_new')
    
    [vals,zvals] = map(lambda a: a.drop([cc for cc in a.coords]),[vals,zvals])
    
    # make density coordinate negative
    if ztype=='density':
        zvals,znew=-1*zvals,-1*znew # trick so we don't need to rewrite the function
        dnew='isop'
    else: 
        dnew='z'
        
    # loop in z (to improve computing efficiency with dask)
    VALS=[]
    for kk in k:
      
        # create mask: 1 for INDEXES closest to (shallower than) the target depth
        mask_l = -1*(znew.isel(dimz=kk)>zvals)
        # shift mask up, get "1" at INDEXEX below target depth
        mask_l = mask_l-mask_l.shift(dimz=1) # mask for lower points
        # shitf mask down, get "1" at INDEXEX above target deptvals
        mask_u = mask_l.shift(dimz=-1) # mask for upper points

        # apply mask to get original values and depths above and below target depth
        vals_il = (vals*mask_l).sum(dim='dimz')
        vals_iu = (vals*mask_u).sum(dim='dimz')
        zold_il = (zvals*mask_l).sum(dim='dimz')
        zold_iu = (zvals*mask_u).sum(dim='dimz')

        # interpolate linearly to get values at target depth
        ### - add surface values for outcropping
        ### - add bottom values for intersection
        
        vals_i = vals_iu + (vals_il-vals_iu)*(znew.isel(dimz=kk)-zold_iu)/(zold_il-zold_iu)
        
        if interp_boundaries:
              
            mask_s = znew.isel(dimz=kk) < zvals.isel(dimz=-1) # isopycnal outside upper limit (outcropping)--- False is outcropping
            mask_b = znew.isel(dimz=kk) > zvals.isel(dimz=0)  # isopycnal outside lower limit (into topography)--- False is insersection
            # replace outcropping values
            vals_i = xr.where(mask_s, vals_i, vals.isel(dimz=-1))
            # replace topography intersection values
            vals_i = xr.where(mask_b, vals_i, vals.isel(dimz=0))
        
        VALS.append(vals_i)
        
    # add coordinates of new depths to dataarray
    if ztype=='density':
        zvals,znew=-1*zvals,-1*znew # return to orignal density values
    
    VALS = xr.concat(VALS,dim='s_rho_new')
    VALS = VALS.rename({'s_rho_new':dnew})
    
    # rename znew coordinate to z
    znew=znew.rename({'dimz':dnew})
    
    VALS = VALS.assign_coords(z=znew)
    #vals_i = vals_i.assign_coords(z=znew)
    
    return VALS

####################################################
# Calculate CROCO C-grid mean within layer #########
####################################################
def layer_average(vals,zvals,zws,zlims,dens=None,ztype='depth'):
    
    '''
    INPUT:
        vals : quantity to average
        zvals: vertical coordinate (depth of rho points)
        zws  : vertical z-grid with cells upper/lower bounds (depht of w points)
        zlims: list or array with the upper and lower limits of the layer
        dens : density values (use with ztype='density') 
        ztype: 'depth' for standard z-level layer or 'density' for isopycnal layer
        
    OUTPUT:
        mean_vals: averaged quantity within the layer delimited by zlims
        
    Dante Napolitano: dante.napolitano@univ-brest.fr
    '''
    
    # make zlims np.array
    zlims=np.array(zlims) if type(zlims) is not np.array else zlims
    
    zlims=np.sort(zlims) # deeper zlevel first (z < 0), shallower isopycnal first (rho>0)
        
    if ztype=='depth':
        # clean coordinates of all files
        [vals,zvals,zws] = map(lambda a: a.drop([cc for cc in a.coords]),[vals,zvals,zws])

    elif ztype=='density':
        # clean coordinates of all files
        [vals,zvals,zws,dens] = map(lambda a: a.drop([cc for cc in a.coords]),[vals,zvals,zws,dens])
        
        # change density lims to zlims
        vbounds=interpolate_vertically(zvals,dens,zlims,dimz='s_rho',ztype='density')
        zlims=None
        zlims = [vbounds.isel(isop=1), # deeper
                 vbounds.isel(isop=0)] # shallower

    # FIND INDEXES CLOSER TO LAYER
    closest_zgmin = ((zvals-zlims[1])**2).argmin('s_rho')
    closest_zgmax = ((zvals-zlims[0])**2).argmin('s_rho')

    indexes=xr.concat([closest_zgmin,closest_zgmax],dim='bound')
    indexes=indexes.isel(isop=0).compute() if 'isop' in indexes.dims else indexes.compute()

    getnear = lambda z,ind: z[ind]

    zglayer = xr.apply_ufunc(getnear, # func
                    zvals,# arg2
                    indexes, # arg3
                    input_core_dims=[['s_rho'],['bound']],
                    output_core_dims=[['bound']],
                    exclude_dims=set(('s_rho',)),
                    vectorize=True,
                    )
    
    # mask values outside the layer (with the first point outside)
    condz = (zvals<=zglayer.isel(bound=0))&(zvals>=zglayer.isel(bound=1))
    condz=condz.chunk({'s_rho':1})

    vals = vals.where(condz)

    # use zw to integrate
    zws=zws.rename({'s_w':'s_rho'})
    
    # integrate vals in the layer
    int_vals = (vals*zws.diff('s_rho')).sum('s_rho')

    #------
    # remove the portion outside the layer
    zgmin = zws.where(zws>=zglayer.isel(bound=0))
    zgmin = zgmin.bfill('s_rho').isel(s_rho=0)
    zgmin = xr.where(~np.isnan(zgmin),zgmin,zvals.isel(s_rho=-1))

    zgmax = zws.where(zws<=zglayer.isel(bound=1))
    zgmax = zgmax.ffill('s_rho').isel(s_rho=-1)
    zgmax = xr.where(~np.isnan(zgmax),zgmax,zvals.isel(s_rho=0))
    
    # project first points outside the layer and multiply by portion of the grid cell intercepted by the layer
    Vmin = vals.ffill('s_rho').isel(s_rho=-1)* (zgmin - zlims[1]) # a portion above the layer (positive)
    Vmax = vals.bfill('s_rho').isel(s_rho=0)* (zlims[0] - zgmax) # a portion below the layer (positive)

    zdiff = xr.DataArray(np.diff(zlims),dims='s_rho') if ztype=='depth' else zlims[1]-zlims[0]
    zdiff = zdiff.where(zdiff>0)
    
    mean_vals = (int_vals - Vmin - Vmax) / zdiff
    
    return mean_vals
 
#######################################################
# Transfer a 2 or 3-D field at rho points to u points #
#######################################################
def rho2u(var,ds=None,coords=False,dimx='xi_'):
    
    '''
    INPUT:
        ds  : croco dataset
        var : name of variable in ds (str) or variable (dataarray) in rho grid
        dimx: name of x dimension (xi_ or x_)
    OUTPUT:
        var_u: variable interpolated onto u grid
    '''
    
    # get variable in rho grid
    var_rho = ds[var] if type(var) is str else var; var=None

    # replace rho-dimension by dummy dimx
    var_rho = var_rho.rename({dimx+'rho':'dimx'})
    
    # removing rho coords
    var_rho = var_rho.drop([coord for coord in var_rho.coords])
    
    # interpolating to u grid
    var_u = (var_rho+var_rho.shift(dimx=-1))*.5; var_rho=None
    # 0.5*(var_rho.isel(dimx=slice(None,-1)) + var_rho.isel(dimx=slice(1,None)))
    var_u = var_u.isel(dimx=slice(None,-1)) # remove nan from shift
    
    # var_u = var_u.transpose('s_rho','eta_rho','dimx')
    
    # adding u coords (assumes _u  is from the same grid as _rho)
    var_u = var_u.rename({'dimx':dimx+'u'})
    
    if coords:
        var_u = var_u.assign_coords(lon_u=ds.lon_u,lat_u=ds.lat_u)
    
    return var_u

#######################################################
# Transfer a 2 or 3-D field at rho points to v points #
#######################################################
def rho2v(var,ds=None,coords=False,dimy='eta_'):
    
    '''
    INPUT:
        ds  : croco dataset
        var : name of variable in ds (str) or variable (dataarray) in rho grid
        dimy: name of y dimension (eta_ or y_)
    OUTPUT:
        var_v: variable interpolated onto v grid
    '''

    # get variable in rho grid
    var_rho = ds[var] if type(var) is str else var; var=None

    # replace rho-dimension by dummy dimy
    var_rho = var_rho.rename({dimy+'rho':'dimy'})
    
    # removing rho coords
    var_rho = var_rho.drop([coord for coord in var_rho.coords])
    
    # interpolating to v grid
    var_v = (var_rho+var_rho.shift(dimy=-1))*.5; var_rho=None
    # = 0.5*(var_rho.isel(dimy=slice(None,-1)) + var_rho.isel(dimy=slice(1,None)))
    var_v = var_v.isel(dimy=slice(None,-1)) # remove nan from shift
    
    # adding v coords (assumes _v is from the same grid as _rho)
    var_v = var_v.rename({'dimy':dimy+'v'})
    
    if coords:
        var_v = var_v.assign_coords(lon_v=ds.lon_v,lat_v=ds.lat_v)
    
    return var_v

#########################################################
# Transfer a 2 or 3-D field at rho points to psi points #
#########################################################
def rho2psi(var,ds=None,coords=False,dimx='xi_',dimy='eta_'):
    
    '''
    INPUT:
        ds  : croco dataset
        var : name of variable in ds (str) or variable (dataarray) in rho grid
        dimx: name of x dimension (xi_ or x_)
        dimy: name of y dimension (eta_ or y_)
    OUTPUT:
        var_psi: variable interpolated onto psi grid
    '''

    # get variable in rho grid
    var_rho = ds[var] if type(var) is str else var; var=None
    
    # replace rho-dimension by dummy dimx and dimy
    var_rho = var_rho.rename({dimx+'rho':'dimx',dimy+'rho':'dimy'})
    
    # removing rho coords
    var_rho = var_rho.drop([coord for coord in var_rho.coords])
    
    # interpolating to psi grid
    var_psi = (var_rho + var_rho.shift(dimy=-1,dimx=-1))*.5; var_rho=None
    var_psi = var_psi.isel(dimy=slice(None,-1),dimx=slice(None,-1))
    # var_psi = 0.25*(var_rho.isel(dimy=slice(1,None),dimx=slice(1,None)) +
    #                 var_rho.isel(dimy=slice(1,None),dimx=slice(None,-1)) +
    #                 var_rho.isel(dimy=slice(None,-1),dimx=slice(None,-1)) +
    #                 var_rho.isel(dimy=slice(None,-1),dimx=slice(1,None)))
    
    # adding psi coords (assumes _psi  is from the same grid as _rho)
    var_psi = var_psi.rename({'dimx':dimx+'u','dimy':dimy+'v'})
    
    if coords:
        var_psi = var_psi.assign_coords(lon_psi=ds.lon_psi,lat_psi=ds.lat_psi)    
    
    return var_psi

#######################################################
# Transfer a 2 or 3-D field at u points to rho points #
#######################################################
def u2rho(var,ds=None,coords=False,dimx='xi_'):
    
    '''
    INPUT:
        ds  : croco dataset
        var : name of variable in ds (str) or variable (dataarray) in u grid
        dimx: name of x dimension (xi_ or x_)
    OUTPUT:
        var_rho: variable interpolated to rho grid
    '''
    
    # get variable in u grid
    var_u = ds[var] if type(var) is str else var; var=None
    
    # replace rho-dimension by dummy dimx
    var_u = var_u.rename({dimx+'u':'dimx'})
    
    # removing rho coords
    var_u = var_u.drop([coord for coord in var_u.coords])
    
    # interpolating to rho grid
    var_rho = (var_u + var_u.shift(dimx=-1))*0.5; var_u=None
    # = 0.5*(var_u.isel(dimx=slice(None,-1)) + var_u.isel(dimx=slice(1,None)))
    var_rho = var_rho.isel(dimx=slice(None,-1)) # remove nan from shift
 
    # copying points to match rho-grid size
    var_rho = xr.concat([var_rho.isel(dimx=0), var_rho, var_rho.isel(dimx=-1)],dim='dimx')    
    
    # adding rho coords (assumes _rho  is from the same grid as _u)
    var_rho = var_rho.rename({'dimx':dimx+'rho'})
    
    if coords:
        var_rho = var_rho.assign_coords(lon_rho=ds.lon_rho,lat_rho=ds.lat_rho)
    
    return var_rho

#######################################################
# Transfer a 2 or 3-D field at v points to rho points #
#######################################################
def v2rho(var,ds=None,coords=False,dimy='eta_'):
    
    '''
    INPUT:
        ds  : croco dataset
        var : name of variable in ds (str) or variable (dataarray) in v grid
        dimy: name of x dimension (eta_ or y_)
    OUTPUT:
        var_rho: variable interpolated onto rho grid
    '''
    
    # get variable in v grid    
    var_v = ds[var] if type(var) is str else var; var=None

    # replace rho-dimension by dummy dimy
    var_v = var_v.rename({dimy+'v':'dimy'})
    
    # removing rho coords
    var_v = var_v.drop([coord for coord in var_v.coords])
    
    # interpolating to rho grid
    var_rho = (var_v + var_v.shift(dimy=-1))*0.5; var_v=None
    # = 0.5*(var_v.isel(dimy=slice(None,-1)) + var_v.isel(dimy=slice(1,None)))
    var_rho = var_rho.isel(dimy=slice(None,-1)) # remove nan from shift
                   
    # copying points to match rho-grid size
    var_rho = xr.concat([var_rho.isel(dimy=0), var_rho, var_rho.isel(dimy=-1)],dim='dimy')    
    
    # adding rho coords (assumes _rho  is from the same grid as _v)
    var_rho = var_rho.rename({'dimy':dimy+'rho'})
    
    if coords:
        var_rho = var_rho.assign_coords(lon_rho=ds.lon_rho,lat_rho=ds.lat_rho)
    
    return var_rho

#########################################################
# Transfer a 2 or 3-D field at psi points to rho points #
#########################################################
def psi2rho(var,ds=None,coords=False,dimx='xi_',dimy='eta_'):
    
    '''
    INPUT:
        ds  : croco dataset
        var : name of variable in ds (str) or variable (dataarray) in psi grid
        dimx: name of x dimension (xi_ or x_)
        dimy: name of y dimension (eta_ or y_)
    OUTPUT:
        var_rho: variable interpolated onto rho grid
    '''

    # get variable in psi grid
    var_psi = ds[var] if type(var) is str else var; var=None
    
    # replace psi-dimension by dummy dimx and dimy
    var_psi = var_psi.rename({dimx+'u':'dimx',dimy+'v':'dimy'})
    
    # removing psi coords
    var_psi = var_psi.drop([coord for coord in var_psi.coords])
    
    # interpolating to rho grid
    var_rho = (var_psi + var_psi.shift(dimy=-1,dimx=-1))*.5; var_psi=None
    var_rho = var_rho.isel(dimy=slice(None,-1),dimx=slice(None,-1))
    # var_rho = 0.25*(var_psi.isel(dimy=slice(1,None),dimx=slice(1,None)) +
    #                 var_psi.isel(dimy=slice(1,None),dimx=slice(None,-1)) +
    #                 var_psi.isel(dimy=slice(None,-1),dimx=slice(None,-1)) +
    #                 var_psi.isel(dimy=slice(None,-1),dimx=slice(1,None)))

    # copying points to match rho-grid size
    var_rho = xr.concat([var_rho.isel(dimx=0), var_rho, var_rho.isel(dimx=-1)],dim='dimx')
    var_rho = xr.concat([var_rho.isel(dimy=0), var_rho, var_rho.isel(dimy=-1)],dim='dimy')
    
    # adding rho coords (assumes _rho  is from the same grid as _psi)
    var_rho = var_rho.rename({'dimx':dimx+'rho','dimy':dimy+'rho'})
    
    if coords:
        var_rho = var_rho.assign_coords(lon_rho=ds.lon_rho,lat_rho=ds.lat_rho)    
    
    return var_rho


###############################################################
# Compute horizontal derivatives on z-levels curvilinear grid #
###############################################################
def diffx(var,pm,grid='rho',dimx='xi_',coords=True):
    
    '''
    INPUT:
        var: variable in z-levels
        pm: curvilinear coordinate metric in XI (m-1)
        grid: 'rho','u','v','psi' (converts pm to the variable grid)
    OUTPUT:
        dvardx: x-derivative of variable in z-levels
    '''

    # check if pm is in the dataset
#     if 'pm' not in ds:
#         raise ValueError("the dataset must contain the curvilinear metric pm")
    
    # adjust pm to the variable grid. Default grid=='rho'
    # xm = ds.pm if grid=='rho' else (rho2u(ds.pm) if grid=='u' else (rho2v(ds.pm) if grid=='v' else (rho2psi(ds.pm) if grid=='psi' else 0)))
    pm = pm if (grid=='rho') else (rho2u(pm,var,coords=coords) if grid=='u' else (rho2v(pm,var,coords=coords) if grid=='v' else (rho2psi(pm,var,coords=coords) if grid=='psi' else pm*0)))
    if pm.all()==0: raise ValueError("grid must be one of: rho, u, v, psi")

    # replace rho/u-dimension by dummy dimx
    oldgrid = 'u' if (grid=='u')|(grid=='psi') else 'rho'# if (grid=='rho')|(grid=='v')
    [var,pm] = map(lambda da: da.rename({dimx+oldgrid:'dimx'}),[var,pm])
    
    # calculate derivative
    dvardx = (var.shift(dimx=-1)-var)*pm # pm = ds.pm converted to the variable grid
    
    var=None;pm=None
    dvardx = dvardx.isel(dimx=slice(None,-1)) # remove NaN from last position
    
    newgrid = 'rho' if (grid=='u')|(grid=='psi') else 'u'# if (grid=='rho')|(grid=='v')
    dvardx = dvardx.rename({'dimx':dimx+newgrid})
    
    return dvardx

def diffy(var,pn,grid='rho',dimy='eta_',coords=True):
    
    '''
    INPUT:
        var: variable in z-levels
        pn: curvilinear coordinate metric in ETA (m-1)
        grid: 'rho','u','v','psi' (converts pm to the variable grid)
    OUTPUT:
        dvardy: y-derivative of variable in z-levels
    '''

    # adjust pm to the variable grid. Default grid=='rho'
    pn = pn if (grid=='rho') else (rho2u(pn,var,coords=coords) if grid=='u' else (rho2v(pn,var,coords=coords) if grid=='v' else (rho2psi(pn,var,coords=coords) if grid=='psi' else pn*0)))
    if pn.all()==0: raise ValueError("grid must be one of: rho, u, v, psi")

    # replace rho/v-dimension by dummy dimy
    oldgrid = 'v' if (grid=='v')|(grid=='psi') else 'rho'# if (grid=='rho')|(grid=='v')
    [var,pn] = map(lambda da: da.rename({dimy+oldgrid:'dimy'}),[var,pn])
    
    # calculate derivative
    dvardy = (var.shift(dimy=-1)-var)*pn # pn = ds.pn converted to the variable grid
    
    var=None;pn=None
    dvardy = dvardy.isel(dimy=slice(None,-1)) # remove NaN from last position
    
    newgrid = 'rho' if (grid=='v')|(grid=='psi') else 'v'# if (grid=='rho')|(grid=='v')
    dvardy = dvardy.rename({'dimy':dimy+newgrid})
    
    return dvardy

############################################################
# Compute vertical derivatives on z-levels #################
############################################################
def diffz(var,zval,grid='rho',dimz='s_'):
    
    '''
    INPUT:
        var: variable in z-levels
        zval: z-levels
        grid: 'rho','w'
    OUTPUT:
        dvardz: z-derivative of variable in z-levels
    '''

    # replace srho/sw-dimension by dummy dimz
    [var,zval]=map(lambda da: da.rename({dimz+grid:'dimz'}),[var,zval])
    
    # calculate derivative
    [dvar,dz] = map(lambda da: da.shift(dimz=-1) - da,[var,zval]) 
    dvardz = dvar/dz
    
    var=None;zval=None
    dvardz = dvardz.isel(dimz=slice(None,-1)) # remove NaN from last position
    
    newgrid = 'w' if (grid=='rho') else 'rho'
    dvardz = dvardz.rename({'dimz':dimz+newgrid})
    
    return dvardz

##############################################################
# Compute horizontal derivatives on sigma-levels (1st order) #
##############################################################
def diffxi_sig(var,pm,zval,vgrid='rho',hgrid='rho',dimx='xi_',dimz='s_'):
    
    '''
    INPUT:
        var: variable to take the derivative
        pm: xi grid distances
        zval: depth levels 's_rho' or 'w' grid
        hgrid: 'rho', 'u', 'v', or 'psi'
        vgrid: 'rho' or 'w'
        dimx: type of model output 'xi_' or 'x_'
        dimz: type of model output 's_' or 'z_'
        
        OUTPUT:
            dvardxi: x derivative on sigma level
            
        (ouputs on original variable vertical grid )
    '''

    # variable on sigma level
    dvardx = diffx(var,pm,grid=hgrid,dimx=dimx,coords=False)

    # to change coordinate (apply chain rule)
    dzdx = diffx(zval,pm,grid=hgrid,dimx=dimx,coords=False)
    
    dvardz = diffz(var,zval,grid=vgrid,dimz=dimz)
    # inteprolate dvardz to original vertical grid
    if 's_w' in dvardz.dims:
        dvardz = 0.5*(dvardz+dvardz.shift(s_w=+1)).rename({'s_w':'s_rho'})
        dvardz = xr.concat([dvardz,dvardz.isel(s_rho=0)*np.nan],dim='s_rho')
    else:
        dvardz = 0.5*(dvardz+dvardz.shift(s_rho=+1)).rename({'s_rho':'s_w'}) 
        dvardz = xr.concat([dvardz,dvardz.isel(s_w=0)*np.nan],dim='s_w')

    # inteprolate dvardz to derivative horizontal grid
    dvardz = rho2u(dvardz) if (hgrid=='v')|(hgrid=='rho') else dvardz.rename({'xi_u':'xi_rho'}).drop_isel(xi_rho=-1)

    # apply the chain rule dbdx_z = dbdx_s - dbdz*dzdx_s
    dvardxi = dvardx - dvardz * dzdx

    return dvardxi

def diffeta_sig(var,pn,zval,vgrid='rho',hgrid='rho',dimy='eta_',dimz='s_'):
    
    '''
    INPUT:
        var: variable to take the derivative
        pn: eta grid distances
        zval: depth levels 's_rho' or 'w' grid
        hgrid: 'rho', 'u', 'v', or 'psi'
        vgrid: 'rho' or 'w'
        dimy: type of model output 'eta_' or 'y_'
        dimz: type of model output 's_' or 'z_'
        
        OUTPUT:
            dvardeta: y derivative on sigma level
            
        (ouputs on original variable vertical grid )
    '''

    # variable on sigma level
    dvardy = diffy(var,pn,grid=hgrid,dimy=dimy,coords=False)

    # to change coordinate (apply chain rule)
    dzdy = diffy(zval,pn,grid=hgrid,dimy=dimy,coords=False)
    
    dvardz = diffz(var,zval,grid=vgrid,dimz=dimz)
    # inteprolate dvardz to original vertical grid
    if 's_w' in dvardz.dims:
        dvardz = 0.5*(dvardz+dvardz.shift(s_w=+1)).rename({'s_w':'s_rho'})
        dvardz = xr.concat([dvardz,dvardz.isel(s_rho=0)*np.nan],dim='s_rho')
    else:
        dvardz = 0.5*(dvardz+dvardz.shift(s_rho=+1)).rename({'s_rho':'s_w'}) 
        dvardz = xr.concat([dvardz,dvardz.isel(s_w=0)*np.nan],dim='s_w')

    # inteprolate dvardz to derivative horizontal grid
    dvardz = rho2v(dvardz) if (hgrid=='u')|(hgrid=='rho') else dvardz.rename({'eta_v':'eta_rho'}).drop_isel(eta_rho=-1)

    # apply the chain rule dbdx_z = dbdx_s - dbdz*dzdx_s
    dvardeta = dvardy - dvardz * dzdy

    return dvardeta

###########################################################
# Functions to plot croco outputs #########################
###########################################################

def rotuv(u,v,angle,ds=None,grid='psi',units='degrees'):

    '''
    INPUT:
         u,v  : velocities along grid
                [xr.DataArray variables or variable names in Dataset ds]
         angle: angle to rotate velocities
                [xr.DataArray variables or variable names in Dataset ds]
         ds   : xr.Dataset containing u,v, and angle variables (optional)
         grid : grid of the final variables
                [str 'psi' or 'rho']
         units: units of the grid angles:
                [str 'degrees' or 'radians']
         
    OUTPUT:
         urot,vrot: velocities rotated into EW / NS direction
    '''

    # accept angle as DataArray variable or variable in a dataset
    angle = angle if type(angle) is xr.DataArray else ds[angle] if angle in ds else 'error'
    if type(angle) is str: raise ValueError('angle must either be a DataArray object or a variable name in ds')

    # transform angle to radians
    if units=='degrees':
        angles = np.pi*angle/180
        
    # accept u and v as DataArray variables or variables in a dataset
    u = u if type(u) is xr.DataArray else ds[u] if u in ds else 'error'
    v = v if type(v) is xr.DataArray else ds[v] if v in ds else 'error'
    if type(u) is str: raise ValueError('u and v must either be DataArray object or a variable name in ds')

    # put variables to rho grid
    if 'xi_u' in u.dims:
        u = u2rho(u)
    if 'eta_v' in v.dims:        
        v = v2rho(v)
        
    
    # rotate vectors by geometric angle
    urot = u*np.cos(angle) - v*np.sin(angle)
    vrot = u*np.sin(angle) + v*np.cos(angle)

    if grid=='psi':
        urot,vrot = rho2psi(urot),rho2psi(vrot)

    return [urot,vrot]

###########################################################

### UTILITY FUNCTIONS

###########################################################

##############################################
# Slice croco outputs ########################
##############################################
def findCROCO_index(gridfile,limits=[None,None,None,None]):

    '''
    Uses CROCO gridfile to find the nearest indexes of a target region in all grids
    
        INPUT:
            gridfile: croco grid .nc file (str)
            limits: (lonmin,lonmax,latmin,latmax). Reference is the rho-grid (center of the cell)
        OUTPUT:
            inds: (lon_min,lon_max,lat_min,lat_max) index of the region in every grid
    '''
    
    MM = xr.open_dataset(gridfile)
    
    # matrix of lons and lats for the Psi points (corners)
    lons = MM.lon_rho.values.squeeze() # lon_psi
    lats = MM.lat_rho.values.squeeze() # lat_psi

    p1 = (limits[0],limits[2]) # SW corner
    p2 = (limits[1],limits[2]) # SE corner
    p3 = (limits[1],limits[3]) # NE corner
    p4 = (limits[0],limits[3]) # NW corner
    
    # find closest point (argdistnear sintax)
    # ind = [np.argmin(np.sqrt((lons-xx)**2 + (lats-yy)**2)) for xx,yy in zip([point[0]], [point[1]])]
    ind = [np.argmin(np.sqrt((lons-pp[0])**2 + (lats-pp[1])**2)) for pp in [p1,p2,p3,p4]]
    # ind = np.array(ind)

    # x and y indices for lon and lat
    
    iy = np.array([int(ii/lons.shape[1]) for ii in ind])
    ix = ind - iy*lons.shape[1]

    # dict with limits for every grid
    inds = dict(
            rho=[ix.min(),ix.max(),iy.min(),iy.max()],
            u  =[ix.min(),ix.max()-1,iy.min(),iy.max()],
            v  =[ix.min(),ix.max(),iy.min(),iy.max()-1],
            psi=[ix.min(),ix.max()+1,iy.min(),iy.max()+1]
            )
    
    print("inds: [x_min,x_max,y_min,y_max]")
    [print(f"{gname} = {inds[gname]}") for gname in inds.keys()]
    print("\n")
    
#     info="u, v   grids: within cell bounded by psi\n"+\
#          "xi,eta grids: limits outside cell (allows to calculate derivatives for psi-points)\n"
    
    return inds #, info

##################################################
# Visualize croco grid ###########################
##################################################
def visualize_grid(gridfile,pltargs={'npoints':4,'marker':'o','colors':['g','k','r','b'],'marker_size':30,'lw':2}):

    '''
    Uses the CROCO gridfile to plot the distribution of rho, u/xi, v/eta, and psi
    points within a range of points
    '''
    
    npo = 4 if 'npoints' not in pltargs else pltargs['npoints']
    mk = 'o' if 'marker' not in pltargs else pltargs['marker']
    cc = ['g','k','r','b'] if 'colors' not in pltargs else pltargs['colors']
    ms = 30 if 'marker_size' not in pltargs else pltargs['marker_size']
    lw = 2 if 'lw' not in pltargs else pltargs['lw']
    fig = plt.figure() if 'fig' not in pltargs else pltargs['fig']
    
    cgrid = xr.open_dataset(gridfile)
    
    # plot grid around PSI points (corners of cell) and position of the other points
    
    ##### PSI points
    l1=plt.scatter(cgrid.lon_psi[:npo-1,:npo-1],cgrid.lat_psi[:npo-1,:npo-1],c=cc[0],s=ms)
    plt.scatter(cgrid.lon_psi[0,0],cgrid.lat_psi[0,0],edgecolors=cc[0],s=ms+ms/2,marker=mk,lw=lw,facecolors='w')
    plt.plot(cgrid.lon_psi[:npo-1,:npo-1],cgrid.lat_psi[:npo-1,:npo-1],c=cc[0],zorder=1)
    plt.plot(cgrid.lon_psi[:npo-1,:npo-1].T,cgrid.lat_psi[:npo-1,:npo-1].T,c=cc[0],zorder=1)
    
    ##### RHO points
    l2=plt.scatter(cgrid.lon_rho[:npo,:npo],cgrid.lat_rho[:npo,:npo],c=cc[1],s=ms)
    plt.scatter(cgrid.lon_rho[0,0],cgrid.lat_rho[0,0],edgecolors=cc[1],s=ms+ms/2,marker=mk,lw=lw,facecolors='w')
    # plt.plot(lonrho,latrho,c='k'); plt.plot(lonrho.T,latrho.T,c='k') # rho-grid
    
    ##### U points (XI points)
    l3=plt.scatter(cgrid.lon_u[:npo,:npo],cgrid.lat_u[:npo,:npo],c=cc[2],s=ms)    
    plt.scatter(cgrid.lon_u[0,0],cgrid.lat_u[0,0],edgecolors=cc[2],s=ms+ms/2,marker=mk,lw=lw,facecolors='w')
    # plt.plot(cgrid.lon_u[:np,:np],cgrid.lat_u[:np,:np],c=cc[2]); plt.plot(cgrid.lon_u[:np,:np].T,cgrid.lat_u[:np,:np].T,c=cc[2]) # u-grid
    
    ##### V points (ETA points)
    l4=plt.scatter(cgrid.lon_v[:npo,:npo],cgrid.lat_v[:npo,:npo],c=cc[3],s=ms)    
    plt.scatter(cgrid.lon_v[0,0],cgrid.lat_v[0,0],edgecolors=cc[3],s=ms+ms/2,marker=mk,lw=lw,facecolors='w')
    # plt.plot(cgrid.lon_v[:np,:np],cgrid.lat_v[:np,:np],c=cc[3]); plt.plot(cgrid.lon_v[:np,:np].T,cgrid.lat_v[:np,:np].T,c=cc[3]) # v-grid
    
    plt.legend([l1,l2,l3,l4],[r'$\psi$',r'$\rho$',r'$\xi$ (u)',r'$\eta$ (v)'],numpoints=1,ncol=2)

    return 

###########################################################
# Functions for other calculations ########################
###########################################################
def calc_dist(lon,lat,dim='eta_v'):

    """
    INPUT:
        lon, lat: xr.DataArray with dimension "dim" [degrees]
    OUTPUT:
        dist: cumulative distance starting at zero [m]
        
    notes
    -> 1 minute = 1 nautical mile = 1852 m
    -> 1 degree = 60 minutes = 60*1852 m
    
    USAGE:
    
        LON = xr.DataArray([10,11],dim='x')
        LAT = xr.DataArray([0,0],dim='x')
        
        dist = calc_dist(LON,LAT,dim='x')
        
        compare with:
            dist,ang = sw.dist(LAT,LON,units='m')
    """
    
    deg2m_lon = 60*1852*np.cos(np.deg2rad(lat + lat.shift({dim:1}))/2).dropna(dim)
    deg2m_lat = 60*1852
    
    dlon = (deg2m_lon*(lon.data[1:]-lon.data[:-1]))**2
    dlat = (deg2m_lat*(lat.data[1:]-lat.data[:-1]))**2
    
    dist = np.hstack([0,np.cumsum(np.sqrt(dlon + dlat))])

    return dist

###########################################################
# Functions to get Jonathan's ncview cmaps ################
###########################################################

# Stealing Jonathan's colormaps
def nc_colormap(name):
    
    '''
    name is the string used in ncview for the colormap
    colormaps.h files needs to be in should be place in directory 
    named nc_colormaps somewhere in the python path
    
    available colormaps:
    
    3gauss,saw,banded,blu_red,blue_red,bright,bw,default,detail,
    extrema,helix,helix2,hotres,jaisn2,jaisnb,jaisnc,jaisnd,jaison,
    jet,manga,rainbow,roullet,ssec,wheel

    '''
    
    # to read cmap, ignore _r option
    ncname = name[:-2] if name[-2:]=='_r' else name

    # path to file
    path = 'nc_colormaps/colormaps_' + ncname + '.h'

    for dirname in sys.path:
        ncfile = os.path.join(dirname, path)
        try:
            f = open(ncfile, 'r')
            break
        except:
            f=None

    if f==None: print('no colormap file for ' + ncname)       
        
    numbers = f.read().split("{")[1].split("}")[0].split(",")
    
    rgb=[]

    for i in range(len(numbers)//3):
        rgb.append((float(np.int(numbers[3*i])/255.),
                    float(int(numbers[3*i+1])/255.),
                    float(int(numbers[3*i+2])/255.)))

    # apply _r option if desiref
    rgb = rgb[::-1] if name[-2:]=='_r' else rgb
    
    my_cmap = colors.LinearSegmentedColormap.from_list('my_colormap',rgb,256)          
    
    return my_cmap