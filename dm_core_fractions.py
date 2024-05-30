import sys
import dm

run = int(sys.argv[1])
start_snap = int(sys.argv[2])
target_snap = int(sys.argv[3])


basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'

dm.dm_halo_core_fractions(basePath, start_snap, target_snap)