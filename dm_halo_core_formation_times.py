import sys
import dm

run = int(sys.argv[1])
start_snap = int(sys.argv[2])

basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'

# dm.dm_halo_core_formation_time(basePath, start_snap)
dm.dm_halo_core_formation_time_lagr_reg(basePath, start_snap)