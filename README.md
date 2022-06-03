# sim070500.github.io

# Set ssh-key
# Source website:https://cynthiachuang.github.io/Generating-a-Ssh-Key-and-Adding-It-to-the-Github/
# Process
# 1. ssh-keygen -t ed25519 -C "your_email@example.com" ( -t means encryption algorithm, ed25519/rsa are both usable )
# 2. cat ~/.ssh/github_key.pub 
# 3. past the key to the github

# Push via ssh
# Process
# 1. git add --all
# 2. git commit -m "Initial commit"
# 3. git remote set-url origin git@github.com:Username/Project.git 
# 4. git branch -M main
# 5. git push -u origin main 
