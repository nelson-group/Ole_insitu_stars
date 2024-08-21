import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py

import funcs
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle

import sys
from os.path import isfile, isdir

plt.style.use('fancy_plots2.mplstyle')

# function to get time difference in Gyr of two scale factors
# adapted from AREPO code
def get_time_difference_in_Gyr(a0, a1):
    omega_lambda = 0.6911
    omega_m = 0.3089
    h = 0.6774
    H0 = 3.2407789e-18 #in s^-1
    
    SEC_PER_MEGAYEAR = 1000000 * 365 * 86400 
    
    factor1 = 2.0 / (3.0 * np.sqrt(omega_lambda))

    term1   = np.sqrt(omega_lambda / omega_m) * a0**1.5
    term2   = np.sqrt(1 + omega_lambda / omega_m * a0**3)
    factor2 = np.log(term1 + term2)

    t0 = factor1 * factor2

    term1   = np.sqrt(omega_lambda / omega_m) * a1**1.5
    term2   = np.sqrt(1 + omega_lambda / omega_m * a1**3)
    factor2 = np.log(term1 + term2)

    t1 = factor1 * factor2

    result = t1 - t0

    time_diff = result / (H0 * h) #now in seconds
    time_diff /= SEC_PER_MEGAYEAR * 1000 #now in gigayears

    return -time_diff

run = int(sys.argv[1])
basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'

# specify directory to save plots
dirname = 'plots/formation_times'
assert isdir(dirname), 'Directory does not exist!'

# specify number of bins to plot:
numBins = 20

assert isfile(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/sub_form_redshifts.hdf5'), 'In-situ stellar content formation redshifts file does not exist!'
f = h5py.File(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/sub_form_redshifts.hdf5','r')
form_snap = f['formation_redshifts_r_vir'][:,:,:]
f.close()

assert isfile('files/' + basePath[32:39] + '/halo_formation_times_SubhaloMassInRadType.hdf5'), 'Halo core formation times file does not exist!'
f = h5py.File('files/' + basePath[32:39] + '/halo_formation_times_SubhaloMassInRadType.hdf5','r')
halo_z_form_2shmr = f['formation_redshift'][:]
f.close()

assert isfile('files/' + basePath[32:39] + '/halo_formation_times.hdf5'), 'Halo formation times file does not exist!'
f = h5py.File('files/' + basePath[32:39] + '/halo_formation_times.hdf5','r')
halo_z_form = f['formation_redshift'][:]
f.close()

assert isfile('files/' + basePath[32:39] + '/galaxy_ages_centrals.hdf5'), 'Galaxy ages file does not exist!'
f = h5py.File('files/' + basePath[32:39] + '/galaxy_ages_centrals.hdf5','r')
galaxy_form_z = f['galaxy_stellar_formation_redshift'][:]
f.close()

assert isfile('files/' + basePath[32:39] + '/galaxy_ages_centrals_insitu.hdf5'), 'In-situ galaxy ages file does not exist!'
f = h5py.File('files/' + basePath[32:39] + '/galaxy_ages_centrals_insitu.hdf5','r')
galaxy_form_z_insitu = f['galaxy_stellar_formation_redshift'][:]
f.close()

file = f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/subhaloFlag_insitu.hdf5'
assert isfile(file), 'Subhalo flag file does not exist!'
f = h5py.File(file,'r')
flag = f['subhaloFlag'][:]
f.close()

h_const = il.groupcat.loadHeader(basePath,99)['HubbleParam']
boxSize = il.groupcat.loadHeader(basePath,99)['BoxSize']

groups = il.groupcat.loadHalos(basePath, 99, fields = ['Group_M_Crit200','GroupFirstSub'])
group_masses = groups['Group_M_Crit200']*1e10/h_const
gfs = groups['GroupFirstSub'][:]

sub_ids, _, halo_ids = np.intersect1d(np.nonzero(flag)[0], gfs, return_indices=True)

group_masses = np.log10(group_masses[halo_ids])

fig, ax = plt.subplots(figsize=(16,9))
styles = ['solid','dotted','dashed']

#insitu stellar content formation redshift
used_halo_masses = group_masses
halo_mass_cutoff = np.where(used_halo_masses > 8)[0]
used_halo_masses = used_halo_masses[halo_mass_cutoff]

print(halo_mass_cutoff.shape)

for ptype in range(3):
    form_snap_masked = form_snap[sub_ids,0,ptype]
    
    form_snap_masked = form_snap_masked[halo_mass_cutoff]
    print(np.where(np.isfinite(form_snap_masked))[0].shape)
    finite_z_mask = np.where(np.isfinite(form_snap_masked))[0]
    print(finite_z_mask.shape)
    group_masses_masked = np.take(used_halo_masses,finite_z_mask,mode='clip')
    insitu_content_form_z = form_snap_masked[finite_z_mask]
    print(insitu_content_form_z)

    xmed,ymed,_,_ = funcs.binData_med(group_masses_masked,insitu_content_form_z, numBins)
    plt.plot(xmed,ymed,color='tab:blue',label = 'in-situ stellar content',linestyle=styles[ptype])

#galaxy ages (all stars)
galaxy_form_z_masked = np.take(galaxy_form_z,halo_ids,mode='clip')
galaxy_form_z_masked = galaxy_form_z_masked[halo_mass_cutoff]
finite_z_mask = np.where(np.isfinite(galaxy_form_z_masked))[0]
group_masses_masked = used_halo_masses[finite_z_mask]
galaxy_form_z_masked = galaxy_form_z_masked[finite_z_mask]

xmed2,ymed2,_,_ = funcs.binData_med(group_masses_masked,galaxy_form_z_masked, numBins)
plt.plot(xmed2,ymed2,color='tab:orange')

#galaxy ages (only in-situ stars)
galaxy_form_z_insitu_masked = np.take(galaxy_form_z_insitu,halo_ids,mode='clip')
galaxy_form_z_insitu_masked = galaxy_form_z_insitu_masked[halo_mass_cutoff]
finite_z_mask = np.where(np.isfinite(galaxy_form_z_insitu_masked))[0]
group_masses_masked = used_halo_masses[finite_z_mask]
galaxy_form_z_insitu_masked = galaxy_form_z_insitu_masked[finite_z_mask]

xmed3,ymed3,_,_ = funcs.binData_med(group_masses_masked,galaxy_form_z_insitu_masked, numBins)
plt.plot(xmed3,ymed3,color='tab:red')

#DM halo formation redshift
halo_z_form_masked = np.take(halo_z_form,halo_ids,mode='clip')
halo_z_form_masked = halo_z_form_masked[halo_mass_cutoff]
finite_z_mask = np.where(np.isfinite(halo_z_form_masked))[0]
group_masses_masked = used_halo_masses[finite_z_mask]
halo_z_form_masked = halo_z_form_masked[finite_z_mask]

xmed4,ymed4,_,_ = funcs.binData_med(group_masses_masked,halo_z_form_masked, numBins)
plt.plot(xmed4,ymed4,color='tab:green')

#DM halo formation redshift; all DM particles within 2SHMR
halo_z_form_2shmr_masked = np.take(halo_z_form_2shmr,halo_ids,mode='clip')
halo_z_form_2shmr_masked = halo_z_form_2shmr_masked[halo_mass_cutoff]
finite_z_mask = np.where(np.isfinite(halo_z_form_2shmr_masked))[0]
group_masses_masked = used_halo_masses[finite_z_mask]
halo_z_form_2shmr_masked = halo_z_form_2shmr_masked[finite_z_mask]

xmed5,ymed5,_,_ = funcs.binData_med(group_masses_masked,halo_z_form_2shmr_masked, numBins)
plt.plot(xmed5,ymed5,color='tab:green', linestyle = 'dashed')

plt.ylim(0.3,3.3)
plt.xlim(10.8,14.0)
plt.xlabel(r'halo mass [$\log\,\rm{M}_\odot$]')
plt.ylabel('formation redshift')

solid = mlines.Line2D([], [], color='C0', linestyle = 'solid',\
                      label = f'all tracers')
dotted = mlines.Line2D([], [], color='C0', linestyle = 'dotted',\
                       label = f'from IGM')
dashed = mlines.Line2D([], [], color='C0', linestyle = 'dashed',\
                       label = f'from mergers')

issc = mlines.Line2D([], [], color='white', linestyle = 'solid', label = 'in-situ stellar content')
gal_age = mlines.Line2D([], [], color='white', linestyle = 'solid', label='galaxy stellar ages')
none = mlines.Line2D([], [], color='white', linestyle = 'solid', label='                    ')
orange = mlines.Line2D([], [], color='C1', linestyle = 'solid',label = 'galaxy stellar age')
red = mlines.Line2D([], [], color='C3', linestyle = 'solid',label = 'in-situ stellar age')
green = mlines.Line2D([], [], color='C2', linestyle = 'solid',label = 'DM halos')
green_dashed = mlines.Line2D([], [], color='C2', linestyle = 'dashed',label = 'DM halos (core)')
empty = mlines.Line2D([], [], color='white', linestyle = 'solid',label = '')

rec_dwarf = Rectangle((10.8,-0.2),0.4,19, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_dwarf)
rec_mw = Rectangle((11.8,-0.2),0.4,19, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_mw)
rec_group = Rectangle((12.6,-0.2),0.8,19, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_group)

group_m = np.array([6.86093282699585, 7.398449420928955, 7.901801109313965, 8.393900871276855, 8.886045455932617,\
                    9.339580535888672, 9.809768676757812, 10.323699951171875, 10.840004920959473, 11.344971656799316,\
                    11.868677139282227, 12.379907608032227, 12.879098892211914, 13.441740036010742, 13.891986846923828])
gal_m = np.array([4.742977619171143, 4.718081474304199, 4.761687278747559, 4.795966625213623, 4.861518859863281,\
                  5.147671699523926, 6.013262748718262, 7.66148042678833, 8.850899696350098, 9.708621978759766,\
                  10.465829849243164, 11.046808242797852, 11.511899948120117, 11.899462699890137, 12.20790958404541])
secax = ax.twiny()
from scipy import interpolate
f_interp = interpolate.interp1d(gal_m, group_m)
new_tick_locations = np.array([9., 9.5, 10., 10.5, 11., 11.5, 12.])
secax.set_xlim(ax.get_xlim())
secax.set_xticks(f_interp(new_tick_locations))
secax.set_xticklabels(new_tick_locations)
secax.minorticks_off()
secax.set_xlabel(r'stellar mass [$\log\,\rm{M}_\odot$]')

triax = ax.twinx()
z_y = np.linspace(0,10,101)
a_y = 1/(1+z_y)
t_y = get_time_difference_in_Gyr(a_y,0)
f_interp = interpolate.interp1d(t_y, z_y)
new_tick_locations = np.array([2,3,4,5,6,7,8])
triax.set_ylim(ax.get_ylim())
triax.set_yticks(f_interp(new_tick_locations))
triax.set_yticklabels(new_tick_locations)
triax.minorticks_off()
triax.set_ylabel('cosmic time [Gyr]')

ax.minorticks_on()

ax.set_xticks([11,12,13,14])
ax.set_xticklabels([11,12,13,14])

leg_1 = plt.legend(handles = [solid,dotted,dashed],ncol=1,loc='upper left', title = 'insitu stellar content', fontsize = 20)
leg_2 = plt.legend(handles = [green,green_dashed, orange, red], ncol = 2, loc = 'upper right')
plt.gca().add_artist(leg_1)
plt.tight_layout()
plt.savefig(dirname + f'/ages_and_formation_times_50-{run}.pdf',format = 'pdf')