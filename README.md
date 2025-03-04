# Tracing the origin of gas fuelling in-situ star formation.

arXiv: https://arxiv.org/abs/2503.00111

### Abstract

We use the TNG50 cosmological magnetohydrodynamical simulation to study the origin and history of in-situ stars. We trace the gas from which these stars form and analyze formation locations, infall times, and several accretion channels. 
Depending on galaxy stellar mass, 30 to 60\% of all classic in-situ stars form outside of the star-forming regions of galaxies. Their formation in massive galaxies is mostly fueled by mergers whereas smooth accretion dominates for low-mass systems.
We find more massive systems to assemble their baryonic matter forming in-situ stars by $z=0$ earlier than lower-mass objects. Only the most massive galaxies in the simulation show a two-phase formation history where in-situ star formation is mostly finished ($>95\%$) at $z=0.6$.
For $z\leq 2$, this baryonic material is spread out to larger distances when considering halos below $10^{12.5}\,\rm{M}_\odot$. For all halo masses above $10^{10.5}\,\rm{M}_\odot$, the DM halo assembled later compared to the baryonic matter content forming in-situ stars.
Between $z=2.5$ and $z=1.5$, most of the gas eventually forming in-situ stars is no longer located in the IGM but the main progenitor. This change occurs earlier in groups than in dwarfs.

### Computational Methods

Since this project is focused on in-situ stars, they will be the target of further explorations. Tracers located in in-situ stars at redshift zero cannot have been associated with black holes before, since tracers can never leave SMBHs. Therefore one only needs to consider tracers in baryonic matter, namely star particles and gas cells.

By matching their particle IDs with the tracers' parent IDs at $z=0$, one can determine the progenitor particle of every in-situ star at all snapshots.
In particular, as a first step, the IDs of the tracers in in-situ stars are selected at $z=0$. They are sorted by stars and subhalos in which the former are located. As the tracer IDs are constant in all snapshots, one can then match the tracer IDs at the desired snapshot with those at $z=0$. After reshuffling to preserve the original order, the tracer IDs are used to infer the corresponding parent particle IDs.

To deal with this information more efficiently, the particle IDs of the tracer parents are cross-matched with the entire subset array of gas and star particles at all snapshots, respectively. The indices into the subset arrays were saved along with the information on whether a particle is a gas cell or a star. This way, one is always able to access a particle's/cell's properties without having to match IDs again. In the following, this is referred to as the 'parent index table'.

Moreover, every particle/cell of interest is located either in a subhalo, inside a halo but not within any subhalo ('inner fuzz'), or in the IGM. The locating function uses pre-computed offsets where the global particle index of the first particle/cell of every subhalo is given, for every particle type. A binary search is performed on these offsets to find the index of the subhalo the particle/cell sits in. By doing so, one obtains the subhalo indices of all the baryonic mass that forms in-situ stars by $z=0$ for every snapshot. This information is stored in the 'subhalo index table'.

### Key Results

Our main findings are:
- Gas forming in-situ stars by $z=0$ enters galaxies of higher mass earlier than low-mass ones. The core of galaxies shows a higher infall time than the outskirts, and especially than the disk if there is one. The radial profiles of the infall times agree with these observations and indicate an inside-out formation scenario. Some tracers appear to have infall times of $z\geq 6$, indicating that they might be part of the Lagrangian region of the galaxy. 
    
- When analyzing the evolution of the fraction of baryonic mass forming in-situ stars by $z=0$, the results are in agreement with the expectations. From redshift $z=2.5$ until $z\approx 1.5$, most of the baryonic matter is within the MP and not in the IGM anymore. This transition happens slightly earlier for more massive objects. It is not instantaneous, since a non-negligible fraction of baryonic material is always located in satellites. However, there is no time when this fraction exceeds 20\%. Below $z\approx 0.6$, the total contribution in mass growth due to mergers exceeds the total accretion from the IGM, albeit only summing up to 10\% of the total mass fraction.
    
- The fraction of gas in galaxies other than the MP is located in systems of similar and lower masses than its $z=0$ hosts. Following the evolution of this mass fraction, one can see the effect of the bottom-up model of cosmic structure formation.

- At the same redshift and normalized to the halo's virial radius, the baryonic matter is spread out further for galaxies in halos with masses between $10^{10}$ and $10^{12.5}\,\rm{M}_\odot$ than for groups. For $z\geq 2$, this trend becomes less clear. The Lagrangian region of groups has not changed much since $z=1$ concerning the radial distribution of material forming in-situ stars by $z=0$. Gas accreted directly from the IGM is distributed to greater distances than gas accreted via mergers. The discrepancy increases for more massive systems and decreases at lower redshifts.

- The in-situ stellar content of a central galaxy assembled earlier than its dark matter halo considering their final masses. They are about equal for $M_{\mathrm{halo}} = 10^{10.5}\,\rm{M}_\odot$. In contrast to the dark matter halo, the in-situ stellar content formation redshift increases with halo mass. Considering the trend, the average stellar age (galaxy age) agrees with the in-situ stellar formation time although being shifted strongly to lower redshifts.

- Classic in-situ stars form from gas accreted freshly from the IGM, from wind-recycled gas, from mergers, and stripped gas from infalling satellites and other halos in similar amounts. For smaller galaxies, accretion from the IGM makes up 60\% (fresh accretion and NEP wind recycling) whereas this fraction drops to 20\% for massive systems. Contrarily, gas accreted from stripping and mergers dominates in-situ star formation there ($\approx 60\%$) and is less important for dwarfs ($\approx 40\%$). We see similar results both for true in-situ and med-situ stars.

![infall times](./pics/mean_infall_times/image_mean_infall_times_50-1.pdf)

## How to work with the data

Many computations need others as a basis. Here is a guide on the correct order to execute the functions in order to reproduce the results stated above.

1. Execute the trace back algorithm.
- Run [parent_index_table.py](./parent_index_table.py) for all snapshots and TNG50 runs you want to do computations on. Modify the filenames for other simulations available on VERA. It takes up quite some storage (~1.5 TB for 8.5*10^8 in-situ star particles). (For comparison reasons, run the same also for ex-situ stars.) 

- If you want to compare to DM (within 2*R_0.5,star), also run [dm_index_table.py](./dm_index_table.py) for all TNG50 runs.

2. Execute algorithms for basic auxilliary catalogs and files.

- Run [create_position_histories.py](./create_position_histories.py) for all TNG50 runs to obtain inter- and extrapolated subhalo positions at all snapshots.

- Run [create_subhalo_sample.py](./create_subhalo_sample.py) to receive a subhaloFlag - array, that is generally used in all computations, funtions, and snapshots.

- Run [sfr_gas_hmr.py](./sfr_gas_hmr.py) for all snapshots and TNG50 runs to use the starforming gas halfmass radius rather than the stellar halfmass radius,

- Run [star_formation_snapshots.py](./star_formation_snapshots.py) to obtain for each tracer the first snapshot in which it is located in a star particle.

- Run [subhalo_index_table.py](./subhalo_index_table.py) for every TNG50 run. You receive information on the subhalo in which each tracer is located at every snapshot. Execute [dm_subhalo_index_table.py](./dm_subhalo_index_table.py) to do the same for the DM particles mentioned earlier. This is also quite expensive in storage (~0.5 TB for 8.5*10^8 in-situ star particles)

- Run [galaxy_ages.py](./galaxy_ages.py) and [halo_form_snap.py](./halo_form_snap.py)

3. Execute advanced algorithms based on step 2., but still auxilliary.

- Run [distance_cats.py](./distance_cats.py) for all snapshots. Given a subhalo sample, it calculates whether each tracer of the sample is within certain distance cuts.

4. Execute higher level code, needs step 3.

- Run [star_formation_distances.py](./star_formation_distances.py). Computes the distance to the subhalo center in units of either R_0.5,star or R_SF,1/2 given the star formation snapshot of each tracer.

5. Execute higher level code, needs step 4.

- Run [StellarAssembly_cat.py](./StellarAssembly_cat.py) to obtain a catalog for (every!) star in the simulation, stating whether it's ex-situ, in-situ, or med-situ. (under construction -> bug detected but not resolved)

- Run [insitu_cat.py](./insitu_cat.py) Same as [insitu_cat.py](./insitu_cat.py) but only stating for all in-situ stars, whether they are in-situ or med-situ.

- Run [infall_and_leaving_times.py](./infall_and_leaving_times.py) to receive the snapshots when tracers entered the galaxy (halo) for the first/last time and left the IGM for the first time.

6. Execute higher level code, needs step 5.

- Run [bin_parents.py](./bin_parents.py) to obtain information on accretion modes and galaxy composition.

- Run [mean_infall_time.py](./mean_infall_time.py) to obtain galaxy radial profiles of the mean tracer infall into the halo as well as individual tracer positions and halo infall times for specific subhalos.

7. Execute higher level code, needs step 6.

- Run [lagrangian_regions_times.py](./lagrangian_regions_times.py) for all snapshots to compute the median tracer distances. Use return_profiles = True and other settings to obtain radial profiles of the tracer fraction for a single or all galaxies.

- Run [baryonic_lagrangian_regions_cat.py](./baryonic_lagrangian_region_cat.py) to compute several measures characterizing the spatial extent of the baryonic lagrangian region of the in-situ stellar content of halos.

8. Execute higher level code, needs step 7.

- Run [insitu_form_snap.py](./insitu_form_snap.py) for retreiving the formation snapshot of the in-situ stellar content of halos.

Now you should have all catalogs to proceed with creating the plots (here the files are displayed in the order of the plots in the paper). Use the following files:

- [plot_in_ex_frac_mass_trend](./plotting_scripts/plot_in_ex_frac_mass_trend.py) for recreating Figure 1 (left). (Needs step: - )

- [plot_MP_frac_evolution.py](./plotting_scripts/plot_MP_frac_evolution.py) for recreating Figure 1 (right). (Needs step: 6 )

- [plot_insitu_frac_profiles](./plotting_scripts/plot_insitu_frac_profiles.py) for recreating Figure 2 (left). (Needs step: -)

- [plot_medsitu_frac_mass_trend.py](./plotting_scripts/plot_medsitu_frac_mass_trend.py) for recreating Figure 2 (right). (Needs step: 4 )

- [plot_mean_infall_times_image.py](./plotting_scripts/plot_mean_infall_times_image.py) for recreating Figure 3. (Needs step: 6 )

- [plot_tracer_frac_rad_prof_evolution.py](./plotting_scripts/plot_tracer_frac_rad_prof_evolution.py) for recreating Figure 4. (Needs step: 7, for cumulative = True and specific single subhalo ID [otherwise modify the code])

- [plot_tracer_frac_evolution.py](./plotting_scripts/plot_tracer_frac_evolution.py) for recreating Figure 5. (Needs step: 6 )

- [plot_other_gal_frac_evolution.py](./plotting_scripts/plot_other_gal_frac_evolution.py) for recreating Figure 6. (Needs step: 6 )

- [plot_cumulative_radial_profiles.py](./plotting_scripts/plot_cumulative_radial_profiles.py) for recreating Figure 7. (Needs step: 7, for cumulative = True)

- [plot_lagr_hmr_mass_trend.py](./plotting_scripts/plot_lagr_hmr_mass_trend.py) for recreating Figure 8. (Needs step: 7, no profiles necessary)

- [plot_formation_times_mass_trend.py](./plotting_scripts/plot_formation_times_mass_trend.py) for recreating Figure 9. (Needs step: 8 )

- [plot_accretion_channels_mass_trend.py](./plotting_scripts/plot_accretion_channels_mass_trend.py) for recreating Figure 10. (Needs step: 6 )

- [plot_specific_time_diffs_mass_trend.py](./plotting_scripts/plot_specific_time_diffs_mass_trend.py) for recreating Figure 11. (Needs step: 5 )

- [plot_bar_lagr_reg_mass_trend.py](./plotting_scripts/plot_bar_lagr_reg_mass_trend.py) for recreating Figure 12. (Needs step: 7 )
