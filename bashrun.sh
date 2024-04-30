#!/bin/bash

# Function to kill all child processes
function kill_children() {
    echo "Received KeyboardInterrupt. Killing all child processes..."
    pkill -P $$  # Kill all child processes of the current process
    exit 1
}

# Trap KeyboardInterrupt signal and call kill_children function
trap kill_children SIGINT

# Define the range for lr and splt
for lr in {1..4}; do
    for splt in {0..2}; do
        # Define the filename with split value, lr value, current time, and date
        filename="output_split${splt}_lr${lr}_$(date +'%Y-%m-%d_%H-%M-%S').txt"
        
        # Run the Python script with the specified parameters and redirect output to the filename
        python modeltrainer.py --tokenizer "cl100k_base" --GPU 0 --lr $lr --splt $splt > $filename &

        # Wait for the current program to finish before starting the next one
        wait
    done
done

# Wait for all child processes to finish
wait
