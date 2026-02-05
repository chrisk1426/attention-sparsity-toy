# Step-by-Step Guide: Transfer Project to Remote Server with NVIDIA GPU

## Prerequisites
- Your remote server address (IP or hostname)
- Your SSH username for the remote server
- SSH access to the remote server (password or SSH key)

---

## STEP 1: Gather Your Server Information

Before starting, you need:
- **Server address**: e.g., `server.example.com` or `123.45.67.89`
- **Username**: Your SSH username on the server
- **Destination path**: Where you want the project (e.g., `~/attention_sparsity_toy`)

**Test your connection:**
```bash
ssh your_username@your_server_address
# If this works, type 'exit' to return to your local machine
```

---

## STEP 2: Prepare Files for Transfer (On Your Local Machine)

Open Terminal and navigate to your project:

```bash
cd /Users/christopherkang/Desktop/26W/Attention_Sparsity_toy
```

**Verify you're in the right directory:**
```bash
ls
# You should see: main.py, requirements.txt, circuit_analysis.py, etc.
```

---

## STEP 3: Transfer Files to Remote Server

### Option A: Using the Transfer Script (Easiest)

1. **Edit the transfer script:**
   ```bash
   nano transfer_to_server.sh
   ```
   Or use any text editor you prefer.

2. **Find and update these 3 lines (around lines 6-8):**
   ```bash
   REMOTE_USER="your_username"        # Change to your SSH username
   REMOTE_HOST="your_server_address"  # Change to your server IP/hostname
   REMOTE_PATH="~/attention_sparsity_toy"  # Change if you want different location
   ```

3. **Save and exit** (in nano: `Ctrl+X`, then `Y`, then `Enter`)

4. **Run the script:**
   ```bash
   ./transfer_to_server.sh
   ```

5. **Enter your password** when prompted (or it will use SSH key if configured)

### Option B: Manual Transfer (If script doesn't work)

**Step 3a: Create directory on remote server**
```bash
ssh your_username@your_server_address "mkdir -p ~/attention_sparsity_toy"
```

**Step 3b: Transfer files**
```bash
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude 'results/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    . your_username@your_server_address:~/attention_sparsity_toy/
```

Replace `your_username` and `your_server_address` with your actual values.

---

## STEP 4: Connect to Remote Server

```bash
ssh your_username@your_server_address
```

You should now be logged into your remote server.

---

## STEP 5: Navigate to Project Directory

```bash
cd ~/attention_sparsity_toy
```

**Verify files transferred correctly:**
```bash
ls -la
# You should see: main.py, requirements.txt, circuit_analysis.py, etc.
```

---

## STEP 6: Check GPU Availability

```bash
nvidia-smi
```

**Expected output:** You should see GPU information. If you get "command not found", the GPU drivers might not be installed (contact your server admin).

**Check PyTorch can see GPU:**
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If this prints `False`, you'll need to install PyTorch with CUDA support (see Step 7).

---

## STEP 7: Set Up Python Environment

### Step 7a: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your prompt.

### Step 7b: Install PyTorch with CUDA Support

**First, check your CUDA version:**
```bash
nvcc --version
# or
cat /usr/local/cuda/version.txt
```

**Then install PyTorch with matching CUDA version:**

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU only (if no GPU):
```bash
pip install torch torchvision torchaudio
```

**Verify PyTorch CUDA:**
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

Should print `CUDA available: True` if everything is set up correctly.

### Step 7c: Install Other Dependencies

```bash
pip install -r requirements.txt
```

This will install: transformers, numpy, scipy, matplotlib, seaborn, pandas, tqdm

---

## STEP 8: Test the Setup

**Run a quick test:**
```bash
python3 -c "
import torch
from circuit_analysis import CircuitAnalyzer
print('✓ All imports successful')
print(f'✓ Device: {\"CUDA\" if torch.cuda.is_available() else \"CPU\"}')
"
```

---

## STEP 9: Run the Main Script

```bash
python3 main.py
```

**What to expect:**
- The script will load the TinyStories-33M model (this may take a few minutes the first time)
- It will compute circuits and perform spectral analysis
- Results will be saved to `results/` directory
- Progress will be printed to console

**First run may take 10-30 minutes** depending on:
- Model download time
- GPU speed
- Number of layers processed

---

## STEP 10: Check Results

After the script completes:

```bash
ls results/
# You should see: *.png files with plots
```

**View results:**
- If you have X11 forwarding: plots will display automatically
- Otherwise: Download the PNG files to view them locally

**Download results to your local machine:**
```bash
# On your LOCAL machine (new terminal, not SSH'd):
scp -r your_username@your_server_address:~/attention_sparsity_toy/results/ ~/Desktop/26W/Attention_Sparsity_toy/results/
```

---

## Troubleshooting Common Issues

### Issue: "Permission denied" when running script
```bash
chmod +x transfer_to_server.sh
```

### Issue: "Connection refused" or "Host unreachable"
- Check server address is correct
- Check you have internet connection
- Verify server is running and accessible

### Issue: "nvidia-smi: command not found"
- GPU drivers may not be installed
- Contact server administrator
- Or use CPU mode (slower but works)

### Issue: "CUDA available: False" after installing PyTorch
- Reinstall PyTorch with correct CUDA version
- Check: `python3 -c "import torch; print(torch.__version__)"`
- Visit https://pytorch.org/get-started/locally/ for exact command

### Issue: "ModuleNotFoundError" when running
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: Out of memory errors
- Process fewer layers at a time
- Reduce batch size in model loading
- Use CPU mode (slower but uses less memory)

---

## Quick Reference Commands

**On Local Machine:**
```bash
cd /Users/christopherkang/Desktop/26W/Attention_Sparsity_toy
./transfer_to_server.sh
```

**On Remote Server:**
```bash
cd ~/attention_sparsity_toy
source venv/bin/activate
python3 main.py
```

**Download results:**
```bash
# From local machine
scp -r user@server:~/attention_sparsity_toy/results/ ./results/
```

---

## Need Help?

If you get stuck at any step:
1. Check the error message carefully
2. Verify you're in the correct directory
3. Make sure virtual environment is activated (if using one)
4. Check GPU is available: `nvidia-smi`

Good luck! 🚀
