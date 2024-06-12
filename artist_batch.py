import subprocess
import os
import time

artist_path = "/import/sw/artist/2.12.6-18-g4550b905-intern-batch/bin-pub/artist-batch"
# artist_path = "/zhome/clarkcs/aRTistLnx/aRTistLnx/bin64/astart.sh"
script_path = "/zhome/clarkcs/artist_scripts/random_objects.tcl"
output_path = "/lhome/clarkcs/aRTist_simulations/aRTist_test_data"

N_SIMULATIONS = 100
for i in range(N_SIMULATIONS):
    start = time.time()
    subprocess.run(["env", f"SIMULATION_NUMBER={i}", f"OUTPUT_PATH={output_path}", artist_path, script_path], check=True)
    print("##############################################")
    print(f"simulation took {time.time() - start} seconds")
    print("##############################################")

print("Done running all simulations")
