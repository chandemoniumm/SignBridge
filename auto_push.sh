#!/bin/bash

# Change to your project directory
cd ~/chandemonium/isl_project

# Stage all changes (new, modified, deleted files)
git add .

# Commit with a timestamped message
git commit -m "Auto-update on $(date +'%Y-%m-%d %H:%M:%S')"

# Push changes to the main branch
git push origin master


