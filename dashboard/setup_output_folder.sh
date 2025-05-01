#!/bin/bash

# Create OUTPUT folder structure if it doesn't exist
mkdir -p public/OUTPUT/boards

# Copy transcript.json to OUTPUT folder
cp ../transcript.json public/OUTPUT/
echo "Copied transcript.json to public/OUTPUT/"

# Copy boards.json to OUTPUT/boards folder
cp ../boards/boards.json public/OUTPUT/boards/
echo "Copied boards.json to public/OUTPUT/boards/"

# Make the script executable with: chmod +x setup_output_folder.sh
# Run with: ./setup_output_folder.sh

echo "Setup complete! The dashboard is now configured to use the OUTPUT folder."
echo "Start the development server with: npm run dev"