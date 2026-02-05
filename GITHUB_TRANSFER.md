# Using GitHub to Transfer Project (Recommended)

## Why Use GitHub?

✅ **Easier**: Push once, pull anywhere  
✅ **Version Control**: Track changes and history  
✅ **Backup**: Your code is safely stored  
✅ **Sync Changes**: Easy to update code on server  
✅ **No File Transfer Issues**: No need for rsync/SCP  

---

## Quick Setup (5 minutes)

### Step 1: Initialize Git (On Your Local Machine)

```bash
cd /Users/christopherkang/Desktop/26W/Attention_Sparsity_toy

# Initialize git repository
git init

# Add all files
git add .

# Make first commit
git commit -m "Initial commit: Spectral Analysis project"
```

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `attention-sparsity-toy` (or any name you like)
3. Make it **Private** (recommended for research projects)
4. **Don't** initialize with README (you already have one)
5. Click "Create repository"

### Step 3: Connect Local to GitHub

GitHub will show you commands. Use these:

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/attention-sparsity-toy.git

# Push to GitHub
git branch -M main
git push -u origin main
```

You'll be prompted for your GitHub username and password (or use a personal access token).

### Step 4: Clone on Remote Server

**SSH into your server:**
```bash
ssh f006j44@lisplab-1.thayer.dartmouth.edu
```

**Clone the repository:**
```bash
cd ~
git clone https://github.com/YOUR_USERNAME/attention-sparsity-toy.git
cd attention-sparsity-toy
```

### Step 5: Set Up on Server (Same as before)

```bash
# Check GPU
nvidia-smi

# Set up Python environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Run!
python3 main.py
```

---

## Updating Code (When You Make Changes)

### On Local Machine:
```bash
cd /Users/christopherkang/Desktop/26W/Attention_Sparsity_toy
git add .
git commit -m "Description of changes"
git push
```

### On Server:
```bash
cd ~/attention-sparsity-toy
git pull
```

That's it! Much easier than transferring files manually.

---

## Comparison: GitHub vs Direct Transfer

| Feature | GitHub | Direct Transfer (rsync/SCP) |
|---------|--------|----------------------------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐ Moderate |
| **Version Control** | ✅ Yes | ❌ No |
| **Backup** | ✅ Yes | ❌ No |
| **Update Code** | `git pull` | Re-transfer all files |
| **Internet Required** | ✅ Yes (for push/pull) | ✅ Yes (for transfer) |
| **GitHub Account** | ✅ Required | ❌ Not needed |
| **Speed** | Fast (only changes) | Fast (rsync is efficient) |

---

## Recommendation

**Use GitHub if:**
- ✅ You have a GitHub account (or can create one)
- ✅ You want version control
- ✅ You'll be making changes and need to sync
- ✅ You want a backup of your code

**Use Direct Transfer if:**
- ❌ You don't want to use GitHub
- ❌ Server doesn't have git installed
- ❌ You only need to transfer once

---

## Quick Start Commands Summary

**Local Machine:**
```bash
cd /Users/christopherkang/Desktop/26W/Attention_Sparsity_toy
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/attention-sparsity-toy.git
git push -u origin main
```

**Remote Server:**
```bash
ssh f006j44@lisplab-1.thayer.dartmouth.edu
git clone https://github.com/YOUR_USERNAME/attention-sparsity-toy.git
cd attention-sparsity-toy
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python3 main.py
```

---

## Troubleshooting

### "Git not installed on server"
```bash
# On server, check if git is installed
which git

# If not, ask your admin or install:
# sudo apt-get install git  # Ubuntu/Debian
# sudo yum install git      # CentOS/RHEL
```

### "Authentication failed" when pushing
- Use a Personal Access Token instead of password
- Go to GitHub Settings → Developer settings → Personal access tokens
- Create token with `repo` permissions
- Use token as password when pushing

### "Repository not found"
- Check repository name matches
- Check you have access (if private repo)
- Verify GitHub username is correct

---

**I recommend using GitHub - it's simpler and gives you version control!** 🚀
