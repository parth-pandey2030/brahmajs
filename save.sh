#!/bin/bash

# Export changes
git add .
git commit -m "$(date)"
git push origin

# Import changes
git pull origin

# Generate changelog
rm CHANGELOG.md
git log --pretty="- %s" > CHANGELOG.md