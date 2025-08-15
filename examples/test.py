from pyrep import PyRep
from rvo2 import ORCA_agent as orca_agent

pr = PyRep()
# Launch the application with a scene file in headless mode
pr.launch('RL_drone_field_10x10.ttt', headless=False) 
pr.start()  # Start the simulation

# Do some stuff

pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application