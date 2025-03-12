echo "# Ad Campaign Optimization Tool" > README.md
echo "venv/" > .gitignore  # Ignore virtual environments
echo "node_modules/" >> .gitignore  # Ignore unnecessary files
git add README.md .gitignore
git commit -m "Added README and gitignore"
git push origin main
