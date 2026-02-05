# Transferring Project to Remote Server with NVIDIA GPU

## Quick Transfer Methods

### Method 1: Using the Transfer Script (Easiest)

1. **Edit the transfer script:**
   ```bash
   nano transfer_to_server.sh
   # or
   vim transfer_to_server.sh
   ```

2. **Update these variables:**
   - `REMOTE_USER`: Your username on the remote server
   - `REMOTE_HOST`: Server address (IP or hostname)
   - `REMOTE_PATH`: Destination directory path

3. **Make script executable and run:**
   ```bash
   chmod +x transfer_to_server.sh
   ./transfer_to_server.sh
   ```

### Method 2: Using SCP (Simple)

```bash
# Replace with your details
USER="your_username"
HOST="your_server_address"
REMOTE_DIR="~/attention_sparsity_toy"

# Create directory on remote server
ssh ${USER}@${HOST} "mkdir -p ${REMOTE_DIR}"

# Transfer all files
scp -r *.py *.txt *.md ${USER}@${HOST}:${REMOTE_DIR}/
```

### Method 3: Using rsync (Recommended - Efficient)

```bash
# Replace with your details
USER="your_username"
HOST="your_server_address"
REMOTE_DIR="~/attention_sparsity_toy"

# Transfer files (excludes cache and results)
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude 'results/' \
    --exclude '*.pyc' \
    . ${USER}@${HOST}:${REMOTE_DIR}/
```

### Method 4: Using Git (If you have a repo)

```bash
# On local machine
git init
git add .
git commit -m "Initial commit"

# Push to remote (GitHub/GitLab) or use git bundle
# Then on remote server:
git clone <your-repo-url>
```

## Setup on Remote Server

After transferring files, SSH into your server:

```bash
ssh your_username@your_server_address
cd ~/attention_sparsity_toy  # or your chosen directory
```

### 1. Check GPU Availability

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Check PyTorch CUDA support
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ for the right command
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3. Run the Project

```bash
# Make sure you're in the project directory
python main.py
```

## File Transfer Checklist

Files to transfer:
- ✅ `*.py` - All Python source files
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - Documentation
- ✅ `project_outline_text.txt` - Project outline
- ❌ `__pycache__/` - Python cache (exclude)
- ❌ `results/` - Output files (exclude, will be regenerated)
- ❌ `venv/` or `env/` - Virtual environments (exclude)

## Troubleshooting

### Permission Issues
```bash
# Make sure you have write permissions on remote server
ssh user@host "chmod -R 755 ~/attention_sparsity_toy"
```

### Connection Issues
```bash
# Test SSH connection first
ssh your_username@your_server_address

# If using key-based auth, make sure your key is added
ssh-copy-id your_username@your_server_address
```

### GPU Not Detected
```bash
# On remote server, check:
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118
```

## Example Complete Workflow

```bash
# 1. On local machine - transfer files
rsync -avz --exclude '__pycache__' --exclude 'results/' \
    . user@server:~/attention_sparsity_toy/

# 2. SSH into server
ssh user@server

# 3. On server - setup
cd ~/attention_sparsity_toy
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 4. Run
python main.py
```
