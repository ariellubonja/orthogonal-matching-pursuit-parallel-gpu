#!/bin/bash

# Script to manage P-cores only configuration on Intel Core Ultra 9 185H
# Toggles hyperthreading, E-cores, and turbo boost together

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Track whether the CPU is supported
SUPPORTED_CPU=0

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Verify CPU model (non-185H: warn and skip without error)
verify_cpu() {
    local cpu_model
    cpu_model=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs)
    if [[ "$cpu_model" == "Intel(R) Core(TM) Ultra 9 185H" ]]; then
        SUPPORTED_CPU=1
    else
        SUPPORTED_CPU=0
        print_warn "CPU model is '$cpu_model', not Intel Core Ultra 9 185H. Skipping; no changes will be made."
    fi
}

# Get current CPU configuration
show_status() {
    print_info "Current CPU Configuration:"
    
    # Count online CPUs
    local online_cpus
    online_cpus=$(grep -c 1 /sys/devices/system/cpu/cpu*/online 2>/dev/null || echo "0")
    local total_cpus
    total_cpus=$(nproc --all)
    echo "  Online CPUs: $online_cpus / $total_cpus"
    
    # Check turbo boost status
    local turbo_status="Unknown"
    if [[ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
        local no_turbo
        no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
        turbo_status=$([[ "$no_turbo" == "0" ]] && echo "Enabled" || echo "Disabled")
    fi
    echo "  Turbo Boost: $turbo_status"
    
    # Show which CPUs are online
    echo -n "  Active CPUs: "
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        cpu_num=$(basename "$cpu" | sed 's/cpu//')
        if [[ -f "$cpu/online" ]]; then
            online=$(cat "$cpu/online" 2>/dev/null || echo "1")
        else
            online="1"  # CPU0 doesn't have online file
        fi
        if [[ "$online" == "1" ]]; then
            echo -n "$cpu_num "
        fi
    done
    echo
}

# Enable P-cores only mode (disable HT, E-cores, and turbo)
enable_pcore_only_mode() {
    print_info "Enabling P-cores only mode (no HT, no E-cores, no turbo)..."
    
    # Disable hyperthreading siblings
    print_info "Disabling hyperthreading siblings..."
    for cpu in 2 4 5 7 9 11; do
        if [[ -f /sys/devices/system/cpu/cpu$cpu/online ]]; then
            echo 0 > /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null || true
        fi
    done
    
    # Disable E-cores
    print_info "Disabling E-cores..."
    for cpu in {12..21}; do
        if [[ -f /sys/devices/system/cpu/cpu$cpu/online ]]; then
            echo 0 > /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null || true
        fi
    done
    
    # Disable turbo boost
    print_info "Disabling turbo boost..."
    if [[ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
        echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
    else
        print_warn "Intel P-state turbo control not available"
    fi
    
    print_info "P-cores only mode enabled (CPUs 0,1,3,6,8,10 active, no turbo)"
}

# Restore full configuration (enable HT, E-cores, and turbo)
restore_full_config() {
    print_info "Restoring full CPU configuration..."
    
    # Re-enable hyperthreading siblings
    print_info "Re-enabling hyperthreading siblings..."
    for cpu in 2 4 5 7 9 11; do
        if [[ -f /sys/devices/system/cpu/cpu$cpu/online ]]; then
            echo 1 > /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null || true
        fi
    done
    
    # Re-enable E-cores
    print_info "Re-enabling E-cores..."
    for cpu in {12..21}; do
        if [[ -f /sys/devices/system/cpu/cpu$cpu/online ]]; then
            echo 1 > /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null || true
        fi
    done
    
    # Re-enable turbo boost
    print_info "Re-enabling turbo boost..."
    if [[ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
        echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
    else
        print_warn "Intel P-state turbo control not available"
    fi
    
    print_info "All cores and turbo boost enabled"
}

# Main function
main() {
    local action=""
    
    # Parse arguments
    case "${1:-}" in
        --disable)
            action="disable"
            ;;
        --enable)
            action="enable"
            ;;
        --status)
            action="status"
            ;;
        --help|-h|"")
            echo "Usage: $0 [--enable|--disable|--status]"
            echo ""
            echo "Options:"
            echo "  --disable  Enable P-cores only mode (disable HT, E-cores, and turbo)"
            echo "  --enable   Restore all cores and turbo boost"
            echo "  --status   Show current CPU configuration"
            echo "  --help     Show this help message"
            echo ""
            echo "Note: On non-Intel Core Ultra 9 185H systems, this script will print a warning and exit without making changes."
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
    
    # Verify CPU first; on non-185H, warn and exit successfully without changes
    verify_cpu
    if [[ "$SUPPORTED_CPU" -ne 1 ]]; then
        # Skip everything, return success
        exit 0
    fi

    # Only require root on supported CPU where we might make changes
    check_root

    # Perform requested action
    if [[ "$action" == "disable" ]]; then
        enable_pcore_only_mode
    elif [[ "$action" == "enable" ]]; then
        restore_full_config
    elif [[ "$action" == "status" ]]; then
        show_status
    fi
}

# Run main function
main "$@"