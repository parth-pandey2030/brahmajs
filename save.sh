#!/bin/bash

# Generate changelog
rm CHANGELOG.md
git log --pretty="- %s" > CHANGELOG.md

# Export changes
git add .
git commit -m "$(date)"
git push origin

# Import changes
git pull origin
