#!/bin/sh

echo "Starting turtlebot data collection for different agent configurations..."

# # Collect data for 3 agents
# echo "Collecting data for 3 turtlebots..."
# python examples/turtlebot_data_collection.py \
# --scenario-name="turtlebot_random_navigation" \
# --max-episodes=10000 \
# --n-agents=3 \
# --field-size=10 \
# --load-type="three"

# echo "3 turtlebots data collection completed."

# Collect data for 4 agents
# echo "Collecting data for 4 turtlebots..."
# python examples/turtlebot_data_collection.py \
# --scenario-name="turtlebot_random_navigation" \
# --max-episodes=10000 \
# --n-agents=4 \
# --field-size=10 \
# --load-type="four"

# echo "4 turtlebots data collection completed."

# Collect data for 6 agents
echo "Collecting data for 6 turtlebots..."
python examples/turtlebot_data_collection.py \
--scenario-name="turtlebot_random_navigation" \
--max-episodes=10000 \
--n-agents=6 \
--field-size=10 \
--load-type="six"

echo "6 turtlebots data collection completed."

echo "All turtlebot data collection completed successfully!" 