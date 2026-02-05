#!/bin/bash
# Script to transfer project files to remote server via SCP

# Configuration - EDIT THESE VALUES
REMOTE_USER="f006j44"
REMOTE_HOST="lisplab-1.thayer.dartmouth.edu"
REMOTE_PATH="~/attention_sparsity_toy"  # Destination directory on remote server
LOCAL_PATH="."  # Current directory (project root)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Transferring Attention Sparsity Project to Remote Server...${NC}"
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo ""

# Create remote directory if it doesn't exist
echo "Creating remote directory..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}"

# Transfer files using rsync (excludes __pycache__, results, etc.)
echo -e "${GREEN}Transferring files...${NC}"
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'results/' \
    --exclude '.git' \
    --exclude '.DS_Store' \
    --exclude 'venv/' \
    --exclude 'env/' \
    ${LOCAL_PATH}/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/

echo -e "${GREEN}Transfer complete!${NC}"
echo ""
echo "Next steps on remote server:"
echo "  1. ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  2. cd ${REMOTE_PATH}"
echo "  3. pip install -r requirements.txt"
echo "  4. python main.py"
