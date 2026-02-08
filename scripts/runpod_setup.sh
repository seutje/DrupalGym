#!/bin/bash
# Setup script for DrupalGym on RunPod (H100 or similar)

set -e

CURRENT_HOME="${HOME:-/root}"

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

COMPOSER_HOME_ABS="$(composer global config home --absolute 2>/dev/null || true)"
if [ -z "$COMPOSER_HOME_ABS" ]; then
  COMPOSER_HOME_ABS="$CURRENT_HOME/.config/composer"
fi
COMPOSER_BIN_DIR="$COMPOSER_HOME_ABS/vendor/bin"
LEGACY_COMPOSER_BIN_DIR="$CURRENT_HOME/.composer/vendor/bin"
export PATH="$COMPOSER_BIN_DIR:$LEGACY_COMPOSER_BIN_DIR:$PATH"

PROFILE_SNIPPET_PATH="/etc/profile.d/drupalgym-composer-path.sh"
cat > "$PROFILE_SNIPPET_PATH" <<EOF
export PATH="$COMPOSER_BIN_DIR:$LEGACY_COMPOSER_BIN_DIR:\$PATH"
EOF
chmod 644 "$PROFILE_SNIPPET_PATH"

CODER_PATH_CONFIG="$COMPOSER_HOME_ABS/vendor/drupal/coder/coder_sniffer"
CODER_PATH_LEGACY="$CURRENT_HOME/.composer/vendor/drupal/coder/coder_sniffer"
if [ -d "$CODER_PATH_CONFIG" ]; then
  phpcs --config-set installed_paths "$CODER_PATH_CONFIG"
elif [ -d "$CODER_PATH_LEGACY" ]; then
  phpcs --config-set installed_paths "$CODER_PATH_LEGACY"
else
  echo "Warning: Drupal coder_sniffer path not found under Composer global directories."
fi

if [ -x "$COMPOSER_BIN_DIR/phpcs" ]; then
  ln -sf "$COMPOSER_BIN_DIR/phpcs" /usr/local/bin/phpcs
elif [ -x "$LEGACY_COMPOSER_BIN_DIR/phpcs" ]; then
  ln -sf "$LEGACY_COMPOSER_BIN_DIR/phpcs" /usr/local/bin/phpcs
fi
if [ -x "$COMPOSER_BIN_DIR/phpstan" ]; then
  ln -sf "$COMPOSER_BIN_DIR/phpstan" /usr/local/bin/phpstan
elif [ -x "$LEGACY_COMPOSER_BIN_DIR/phpstan" ]; then
  ln -sf "$LEGACY_COMPOSER_BIN_DIR/phpstan" /usr/local/bin/phpstan
fi

echo "Verifying PHP tooling..."
php -v | head -n 1
echo "php path: $(command -v php)"
phpcs --version
echo "phpcs path: $(command -v phpcs)"
phpstan --version
echo "phpstan path: $(command -v phpstan)"
phpcs -i

# Install Unsloth for optimized training if possible
# Note: Unsloth installation can be tricky depending on the CUDA version.
# For H100 (sm_90), we use the following:
#pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

echo "Setup complete. You can now run the pipeline."
echo "Use 'source venv/bin/activate' to enter the environment."
echo "PHP tooling installed: php, phpcs, phpstan (Drupal standard configured)."
