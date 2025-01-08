git add .
git commit -m "$(date)"
git push origin

rm CHANGELOG.md
git log --pretty="- %s" > CHANGELOG.md