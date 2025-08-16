#!/bin/sh

echo "Starting data collection for different agent configurations..."

# Collect data for 3 agents
echo "Collecting data for 3 agents..."
python examples/orca_drone_navigation_demo_v0_MBRL.py \
--scenario-name="orca_navigation_drone_MBRL" \
--max-episodes=10000 \
--n-agents=3 \
--field-size=10 \
--load-type="three"

echo "3 agents data collection completed."

# Collect data for 4 agents
echo "Collecting data for 4 agents..."
python examples/orca_drone_navigation_demo_v0_MBRL.py \
--scenario-name="orca_navigation_drone_MBRL" \
--max-episodes=10000 \
--n-agents=4 \
--field-size=10 \
--load-type="four"

echo "4 agents data collection completed."

# Collect data for 6 agents
echo "Collecting data for 6 agents..."
python examples/orca_drone_navigation_demo_v0_MBRL.py \
--scenario-name="orca_navigation_drone_MBRL" \
--max-episodes=10000 \
--n-agents=6 \
--field-size=10 \
--load-type="six"

echo "6 agents data collection completed."

echo "All data collection completed successfully!"


