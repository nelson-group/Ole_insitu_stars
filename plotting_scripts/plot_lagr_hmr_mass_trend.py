import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py
import illustrisFuncs as iF

import funcs
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib as mpl

import sys
from os.path import isfile, isdir

plt.style.use('fancy_plots2.mplstyle')

run = int(sys.argv[1])
basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
h = il.groupcat.loadHeader(basePath, 99)
h_const = h['HubbleParam']

# set snapshots to plot
snaps = np.array([99,84,67,50,33,25])
z = iF.give_z_array(basePath)
z_snaps = np.flip(z)[snaps]

# specify directory to save plots
dirname = 'plots/halfmass_radii'
assert isdir(dirname), 'Directory does not exist!'

groups = il.groupcat.loadHalos(basePath, 99, fields = ['Group_M_Crit200', 'GroupFirstSub'])
group_masses =  groups['Group_M_Crit200'][:]* 1e10/h_const
gfs = groups['GroupFirstSub'][:]
del groups

style = ['solid','dotted','dashed']
all_ptype_snaps = [50]
what_to_plot = 'r_vir'
with mpl.rc_context({'xtick.top' : False}): 
    fig,ax = plt.subplots(1,1, figsize = (16,9))
for i in range(0,snaps.size):
    # set particle type: 0 = all tracers, 1 = gas from IGM, 2 = gas from mergers
    ptype = 0

    assert isfile(f'/vera/ptmp/gc/olwitt/insitu/TNG50-{run}/lagrangian_regions/lagrangian_regions_{snaps[i]}.hdf5'),\
            'Lagrangian regions file does not exist!'
    file = f'/vera/ptmp/gc/olwitt/insitu/TNG50-{run}/lagrangian_regions/lagrangian_regions_{snaps[i]}.hdf5'
    f = h5py.File(file,'r')
    subhaloFlag = f['subhaloFlag'][:]
    hmr = f['lagrangian_regions_'+what_to_plot][:,:]
    hmr = hmr[np.nonzero(subhaloFlag)[0],:]
    f.close()

    #match gfs and positive subhaloFlag
    _, gfs_inds, _ = np.intersect1d(gfs, np.nonzero(subhaloFlag)[0], return_indices = True)
    
    masses = np.log10(group_masses[gfs_inds])

    if snaps[i] in all_ptype_snaps:
        for ptype in range(3):
            xmed, ymed, y16,y84 = funcs.binData_med(masses, hmr[:,ptype],25)
            ax.plot(xmed,ymed,color = 'C'+str(i),linestyle=style[ptype])
            # ax.fill_between(xmed,y16,y84, alpha = 0.2)
    else:
        xmed, ymed, y16,y84 = funcs.binData_med(masses, hmr[:,ptype],25)
        ax.plot(xmed,ymed,color = 'C'+str(i),linestyle='solid')
        if snaps[i] == 33:
            ax.fill_between(xmed,y16,y84, alpha = 0.2)

rec_dwarf = Rectangle((10.8,-0.2),0.4,1900, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_dwarf)
rec_mw = Rectangle((11.8,-0.2),0.4,1900, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_mw)
rec_group = Rectangle((12.6,-0.2),0.8,1900, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_group)
ax.set_xlim(10.8,14)    
ax.set_ylim(0.008,10)
ax.set_yscale('log')

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

ax.minorticks_on()
ax.grid(which = 'major',axis = 'y')
ax.set_xticks([11,12,13,14])
ax.set_xticklabels([11,12,13,14])

solid = mlines.Line2D([], [], color = 'gray', linestyle = 'solid',\
                      label = f'all tracers')
dotted = mlines.Line2D([], [], color = 'gray', linestyle = 'dotted',\
                       label = f'from IGM')
dashed = mlines.Line2D([], [], color = 'gray', linestyle = 'dashed',\
                       label = f'from mergers')

blue = mpatches.Patch(color='C0', linestyle = 'solid', label = f'z = {z_snaps[0]:.1f}')
orange = mpatches.Patch(color='C1', linestyle = 'solid', label = f'z = {z_snaps[1]:.1f}')
green = mpatches.Patch(color='C2', linestyle = 'solid', label = f'z = {z_snaps[2]:.1f}')
red = mpatches.Patch(color='C3', linestyle = 'solid', label = f'z = {z_snaps[3]:.1f}')
purple = mpatches.Patch(color='C4', linestyle = 'solid', label = f'z = {z_snaps[4]:.1f}')
brown = mpatches.Patch(color='C5', linestyle = 'solid', label = f'z = {z_snaps[5]:.1f}')

legend1 = plt.legend(handles=[blue,orange,green,red,purple,brown],ncol=2, loc = 'upper left')
legend2 = plt.legend(handles = [solid,dotted,dashed], loc = 'upper right')
plt.gca().add_artist(legend1)
ax.set_xlabel(r'halo mass [$\log\,\rm{M}_\odot$]')
ax.set_ylabel(r'Lagrangian half-mass radius $R_{\rm L, 1/2}$ [$R_{\rm 200c}$]')
fig.tight_layout()
plt.savefig(dirname + '/hmr_vs_mass_' + what_to_plot + f'_50-{run}.pdf',format='pdf')