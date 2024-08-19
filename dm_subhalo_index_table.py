import sys
import dm

run = int(sys.argv[1])    
    
basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
dm.save_location_dm(basePath)