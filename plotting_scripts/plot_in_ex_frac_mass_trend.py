import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py
import tracerFuncs as tF
import funcs
import matplotlib as mpl
from matplotlib.patches import Rectangle
from os.path import isfile, isdir
import sys

# optional: use individual mpl style file
# plt.style.use('fancy_plots2.mplstyle')

run = int(sys.argv[1])
basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
h_const = il.groupcat.loadHeader(basePath,99)['HubbleParam']
boxSize = il.groupcat.loadHeader(basePath,99)['BoxSize']

# snapshot portrayed in the figure
fig_snap = 99

# load stellar masses and ex-situ stellar masses (from Rodriguez-Gomez et al. 2016)
f = h5py.File(basePath[:-6] + 'postprocessing/StellarAssembly/galaxies_' + str(fig_snap).zfill(3) + '.hdf5','r')
stellar_mass = f['StellarMassTotal'][:] * 1e10 / h_const
exsitu_stellar_mass = f['StellarMassExSitu'][:] * 1e10 / h_const
f.close()

print('masses loaded',flush = True)

# load subhalo flags
assert isfile(f'/vera/ptmp/gc/olwitt/auCats/TNG50-{run}/subhaloFlag_insitu.hdf5')
f = h5py.File(f'/vera/ptmp/gc/olwitt/auCats/TNG50-{run}/subhaloFlag_insitu.hdf5','r')
subhaloFlag = f['subhaloFlag'][:]
f.close()


# either load in-situ, ex-situ, and med-situ flags...

# assert isfile('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + f'/StellarAssembly_{fig_snap}_v2.hdf5')
# f = h5py.File('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + f'/StellarAssembly_{fig_snap}_v2.hdf5','r')
# situ_cat = f['stellar_assembly'][:]
# f.close()

# ... or load star formation distances

assert isfile(f'/vera/ptmp/gc/olwitt/insitu/TNG50-{run}/star_formation_distances.hdf5')
f = h5py.File(f'/vera/ptmp/gc/olwitt/insitu/TNG50-{run}/star_formation_distances.hdf5','r')
star_formation_distances = f['star_formation_distances'][:]
f.close()

print('situ cats loaded',flush = True)

# load offsets for flags (stars in subhalos)

# g = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_' + str(fig_snap).zfill(3) + '.hdf5','r')
# starsInSubOffset = g['Subhalo/SnapByType'][:,4]
# g.close()

# numStarsInSubs = il.groupcat.loadSubhalos(basePath, fig_snap, fields = ['SubhaloLenType'])[:,4]

# print('lengths loaded',flush = True)

# load offsets for starformation distances (in-situ stars in subhalos)
assert isfile(f'/vera/ptmp/gc/olwitt/insitu/TNG50-{run}/parent_indices_99.hdf5')
f = h5py.File(f'/vera/ptmp/gc/olwitt/insitu/TNG50-{run}/parent_indices_99.hdf5','r')
numTracersInParents = f['snap_99/numTracersInParents'][:]
f.close()
insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath,99)
final_offsets = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(int)
starsInSubOffset = np.insert(final_offsets,0,0)

print('offsets loaded',flush = True)

situ_cat = np.zeros(star_formation_distances.shape[0], dtype = np.byte)
situ_cat[np.where(star_formation_distances < 2)[0]] = 1 #insitu = 1, medsitu = 0

# compute med-situ stellar mass for each subhalo

insitu_mass = stellar_mass - exsitu_stellar_mass
medsitu_stellar_mass = np.zeros(stellar_mass.shape[0])
insitu_stellar_mass = np.zeros(stellar_mass.shape[0])
for i in range(subhaloFlag.shape[0]):
    if subhaloFlag[i] == 0:
        continue
    # indices = np.arange(starsInSubOffset[i], starsInSubOffset[i] + numStarsInSubs[i])
    indices = np.arange(starsInSubOffset[i], starsInSubOffset[i+1])
    if indices.shape[0] > 0:
        insitu = np.where(situ_cat[indices] == 1)[0]
        # use 0 below when using star formation distances, and 2 when using StellarAssembly
        medsitu = np.where(situ_cat[indices] == 0)[0] 

        medsitu_stellar_mass[i] = insitu_mass[i] * medsitu.shape[0]/indices.shape[0]
        insitu_stellar_mass[i] = insitu_mass[i] * insitu.shape[0]/indices.shape[0]
print('done',flush = True)


sub_mask = np.nonzero(subhaloFlag)[0]
infracs = insitu_stellar_mass[sub_mask] / stellar_mass[sub_mask]
medfracs = medsitu_stellar_mass[sub_mask] / stellar_mass[sub_mask]
exfracs = exsitu_stellar_mass[sub_mask] / stellar_mass[sub_mask]

# comment out the following line to exclude med-situ stars
infracs = 1 - exfracs

log_stellar_mass = np.log10(stellar_mass[sub_mask])

mass_bins, ex_bins, exy16, exy84 =  funcs.binData_med(log_stellar_mass, exfracs, 20)
_, in_bins, iny16, iny84 =  funcs.binData_med(log_stellar_mass, infracs, 20)
_, med_bins, medy16, medy84 =  funcs.binData_med(log_stellar_mass, medfracs, 20)

# mass_bins, ex_bins =  funcs.binData_mean(log_stellar_mass, exfracs, 20)
# _, in_bins =  funcs.binData_mean(log_stellar_mass, infracs, 20)
# _, med_bins =  funcs.binData_mean(log_stellar_mass, medfracs, 20)

# mask = np.where(stellar_mass > 8.5)

mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22

fig, ax = plt.subplots(1,1, figsize = (16,9))
plt.hist2d(log_stellar_mass, infracs, cmap = 'Blues',norm = mpl.colors.LogNorm(), range = [[9,12],[0,1]],\
            bins = (64,64), rasterized = True)
cb = plt.colorbar()
cb.set_label(label='number of galaxies per bin',size = 22)
ax.plot(mass_bins, ex_bins, color = 'tab:orange', label = 'ex-situ')
ax.plot(mass_bins, in_bins, color = 'tab:green', label = 'in-situ')
ax.plot(mass_bins, med_bins, color = 'tab:red', label = 'med-situ')
# ax.fill_between(mass_bins, exy16,exy84,alpha = 0.3,color = 'tab:orange')
# ax.fill_between(mass_bins, iny16,iny84,alpha = 0.3,color = 'tab:green')
# ax.fill_between(mass_bins, medy16,medy84,alpha = 0.3,color = 'tab:red')

plt.legend(loc = 'center left', bbox_to_anchor = (0.075,0.35))
plt.xlim(9,12)
plt.ylim(-0.03,1.02)

# add second x-axis for halo mass (conversion done with TNG50-1 centrals)
group_m = np.array([6.86093282699585, 7.398449420928955, 7.901801109313965, 8.393900871276855, 8.886045455932617,\
                    9.339580535888672, 9.809768676757812, 10.323699951171875, 10.840004920959473, 11.344971656799316,\
                    11.868677139282227, 12.379907608032227, 12.879098892211914, 13.441740036010742, 13.891986846923828])
gal_m = np.array([4.742977619171143, 4.718081474304199, 4.761687278747559, 4.795966625213623, 4.861518859863281,\
                  5.147671699523926, 6.013262748718262, 7.66148042678833, 8.850899696350098, 9.708621978759766,\
                  10.465829849243164, 11.046808242797852, 11.511899948120117, 11.899462699890137, 12.20790958404541])
secax = ax.twiny()
from scipy import interpolate
f_interp = interpolate.interp1d(group_m, gal_m)
new_tick_locations = np.array([11.,11.5,12.,12.5,13.,13.5])
secax.set_xlim(ax.get_xlim())
secax.set_xticks(f_interp(new_tick_locations))
secax.set_xticklabels(new_tick_locations, size = 20)
secax.minorticks_off()
secax.set_xlabel(r'halo mass [$\log\;\rm{M}_\odot$]', size = 22)

rec_dwarf = Rectangle((f_interp(10.8),-0.1),0.5,1.2, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_dwarf)
rec_mw = Rectangle((f_interp(11.8),-0.1),0.4,1.2, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_mw)
rec_group = Rectangle((f_interp(12.6),-0.1),0.62,1.2, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_group)


ax.set_xlabel(r'stellar mass [$\log\;\rm{M}_\odot$]', size = 22)
ax.set_ylabel('stellar mass fraction', size = 22)
fig.tight_layout()

# specify path to your output directory
dirname = 'pics/ex-situ_in-situ_mass_fraction'
assert isdir(dirname), 'Directory not found'
# plt.savefig(dirname + f'/ex-in-situ_mass_frac_50-{run}.pdf',format = 'pdf')
plt.savefig(dirname + f'/ex-in-situ_mass_frac_50-{run}.jpg',format = 'jpg')