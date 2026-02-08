#!/bin/bash
# Setup script for DrupalGym on RunPod (H100 or similar)

set -e

echo "Updating system and installing dependencies..."
apt-get update && apt-get install -y git python3-pip python3-venv php-cli php-xml composer unzip

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing PHP quality tooling (phpcs + phpstan + Drupal coder)..."
composer global require drupal/coder phpstan/phpstan
export PATH="/root/.config/composer/vendor/bin:/root/.composer/vendor/bin:$PATH"

CODER_PATH_CONFIG="/root/.config/composer/vendor/drupal/coder/coder_sniffer"
CODER_PATH_LEGACY="/root/.composer/vendor/drupal/coder/coder_sniffer"
if [ -d "$CODER_PATH_CONFIG" ]; then
  phpcs --config-set installed_paths "$CODER_PATH_CONFIG"
elif [ -d "$CODER_PATH_LEGACY" ]; then
  phpcs --config-set installed_paths "$CODER_PATH_LEGACY"
else
  echo "Warning: Drupal coder_sniffer path not found under Composer global directories."
fi

echo "Verifying PHP tooling..."
php -v | head -n 1
phpcs --version
phpstan --version
phpcs -i

# Install Unsloth for optimized training if possible
# Note: Unsloth installation can be tricky depending on the CUDA version.
# For H100 (sm_90), we use the following:
#pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

echo "Setup complete. You can now run the pipeline."
echo "Use 'source venv/bin/activate' to enter the environment."
echo "PHP tooling installed: php, phpcs, phpstan (Drupal standard configured)."
