#!/bin/bash

# OpenCog Ecosystem - Unified Build Script
# This script provides unified build orchestration for the entire OpenCog ecosystem

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/build"
INSTALL_PREFIX="${BUILD_ROOT}/install"
PARALLEL_JOBS=${PARALLEL_JOBS:-$(nproc)}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print usage
print_usage() {
    cat << EOF
OpenCog Ecosystem Unified Build Script

Usage: $0 [OPTIONS] [COMPONENT...]

Options:
  -h, --help         Show this help message
  -c, --clean        Clean build directories before building
  -j, --jobs N       Number of parallel jobs (default: $(nproc))
  -p, --prefix PATH  Installation prefix (default: ./build/install)
  -t, --test         Run tests after building
  -v, --verbose      Enable verbose output
  --python-only      Build only Python components
  --cmake-only       Build only CMake components
  --rust-only        Build only Rust components

Components:
  If no components specified, builds all components in dependency order.
  Available component categories: as, ai, bi, ct, dv, em, gm, in, nl, oc, ro, sv, wb

Examples:
  $0                    # Build all components
  $0 -c -t              # Clean build and run tests
  $0 --python-only      # Build only Python components
  $0 as ai              # Build only AtomSpace and AI components
EOF
}

# Component build order based on dependencies
declare -a BUILD_ORDER=(
    "dv/cogutil"
    "as/atomspace"
    "as/atomspace-rocks"
    "as/atomspace-cog"
    "sv/cogserver"
    "ai/ure"
    "ai/pln"
    "ai/learn"
    "nl/link-grammar"
    "nl/relex"
    "ro/sensory"
    "ct/attention"
    "ct/spacetime"
    "wb/python-client"
)

# Initialize build environment
init_build_env() {
    log_info "Initializing build environment..."
    
    # Create build directories
    mkdir -p "${BUILD_ROOT}"
    mkdir -p "${INSTALL_PREFIX}"
    
    # Set environment variables
    export CMAKE_PREFIX_PATH="${INSTALL_PREFIX}:${CMAKE_PREFIX_PATH}"
    export PKG_CONFIG_PATH="${INSTALL_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
    export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}"
    export PYTHONPATH="${INSTALL_PREFIX}/lib/python3.12/site-packages:${PYTHONPATH}"
    
    log_success "Build environment initialized"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    if [ -f "${SCRIPT_DIR}/requirements-consolidated.txt" ]; then
        pip install -r "${SCRIPT_DIR}/requirements-consolidated.txt"
        log_success "Python dependencies installed"
    else
        log_warning "requirements-consolidated.txt not found, using fallback requirements.txt"
        if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
            pip install -r "${SCRIPT_DIR}/requirements.txt"
        fi
    fi
}

# Build CMake component
build_cmake_component() {
    local component_path="$1"
    local component_name=$(basename "$component_path")
    
    log_info "Building CMake component: $component_name"
    
    local component_dir="${SCRIPT_DIR}/orc-${component_path}"
    local build_dir="${BUILD_ROOT}/${component_name}"
    
    if [ ! -d "$component_dir" ]; then
        log_error "Component directory not found: $component_dir"
        return 1
    fi
    
    # Find CMakeLists.txt
    local cmake_file
    if [ -f "${component_dir}/CMakeLists.txt" ]; then
        cmake_file="${component_dir}/CMakeLists.txt"
    else
        # Look for CMakeLists.txt in subdirectories
        cmake_file=$(find "$component_dir" -name "CMakeLists.txt" -type f | head -1)
    fi
    
    if [ -z "$cmake_file" ]; then
        log_warning "No CMakeLists.txt found for $component_name, skipping"
        return 0
    fi
    
    local source_dir=$(dirname "$cmake_file")
    
    mkdir -p "$build_dir"
    cd "$build_dir"
    
    # Configure
    cmake "$source_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX"
    
    # Build
    make -j"$PARALLEL_JOBS"
    
    # Install
    make install
    
    log_success "Built CMake component: $component_name"
}

# Build Python component  
build_python_component() {
    local component_path="$1"
    local component_name=$(basename "$component_path")
    
    log_info "Building Python component: $component_name"
    
    local component_dir="${SCRIPT_DIR}/orc-${component_path}"
    
    if [ ! -d "$component_dir" ]; then
        log_error "Component directory not found: $component_dir"
        return 1
    fi
    
    cd "$component_dir"
    
    # Look for Python setup files
    if [ -f "setup.py" ]; then
        python setup.py install --prefix="$INSTALL_PREFIX"
    elif [ -f "pyproject.toml" ]; then
        pip install -e . --prefix="$INSTALL_PREFIX"
    elif [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        log_warning "No Python build configuration found for $component_name"
    fi
    
    log_success "Built Python component: $component_name"
}

# Build Rust component
build_rust_component() {
    local component_path="$1"  
    local component_name=$(basename "$component_path")
    
    log_info "Building Rust component: $component_name"
    
    local component_dir="${SCRIPT_DIR}/orc-${component_path}"
    
    if [ ! -d "$component_dir" ]; then
        log_error "Component directory not found: $component_dir"
        return 1
    fi
    
    cd "$component_dir"
    
    if [ -f "Cargo.toml" ]; then
        cargo build --release
        cargo install --path . --root "$INSTALL_PREFIX"
    else
        log_warning "No Cargo.toml found for $component_name"
    fi
    
    log_success "Built Rust component: $component_name"
}

# Run tests for component
run_component_tests() {
    local component_path="$1"
    local component_name=$(basename "$component_path")
    
    log_info "Running tests for: $component_name"
    
    local component_dir="${SCRIPT_DIR}/orc-${component_path}"
    
    cd "$component_dir"
    
    # Run CMake tests
    if [ -d "${BUILD_ROOT}/${component_name}" ]; then
        cd "${BUILD_ROOT}/${component_name}"
        if make test 2>/dev/null; then
            log_success "CMake tests passed for $component_name"
        fi
    fi
    
    # Run Python tests
    cd "$component_dir"
    if [ -d "tests" ] || [ -f "test_*.py" ]; then
        python -m pytest . || log_warning "Python tests failed for $component_name"
    fi
    
    # Run Rust tests
    if [ -f "Cargo.toml" ]; then
        cargo test || log_warning "Rust tests failed for $component_name"
    fi
}

# Main build function
build_ecosystem() {
    local components=("$@")
    local run_tests=false
    local python_only=false
    local cmake_only=false
    local rust_only=false
    local clean=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -c|--clean)
                clean=true
                shift
                ;;
            -j|--jobs)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            -p|--prefix)
                INSTALL_PREFIX="$2"
                shift 2
                ;;
            -t|--test)
                run_tests=true
                shift
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            --python-only)
                python_only=true
                shift
                ;;
            --cmake-only)
                cmake_only=true
                shift
                ;;
            --rust-only)
                rust_only=true
                shift
                ;;
            *)
                components+=("$1")
                shift
                ;;
        esac
    done
    
    # Clean if requested
    if [ "$clean" = true ]; then
        log_info "Cleaning build directories..."
        rm -rf "$BUILD_ROOT"
    fi
    
    # Initialize build environment
    init_build_env
    
    # Install Python dependencies
    install_python_deps
    
    # Determine components to build
    local build_components
    if [ ${#components[@]} -eq 0 ]; then
        build_components=("${BUILD_ORDER[@]}")
    else
        build_components=("${components[@]}")
    fi
    
    # Build components
    for component in "${build_components[@]}"; do
        if [ "$python_only" != true ] && [ "$rust_only" != true ]; then
            build_cmake_component "$component"
        fi
        
        if [ "$cmake_only" != true ] && [ "$rust_only" != true ]; then
            build_python_component "$component"  
        fi
        
        if [ "$cmake_only" != true ] && [ "$python_only" != true ]; then
            build_rust_component "$component"
        fi
        
        if [ "$run_tests" = true ]; then
            run_component_tests "$component"
        fi
    done
    
    log_success "OpenCog ecosystem build completed!"
    log_info "Installation directory: $INSTALL_PREFIX"
}

# Main entry point
main() {
    build_ecosystem "$@"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi