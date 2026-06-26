#!/bin/bash

# ================= Configuration =================
TIMEOUT_SEC=10
# Excluded partitions
EXCLUDE_PARTS="defq.*|V100q|NV100q" 

# == Alignment settings ==
# 11 blocks = 22 characters wide
ALIGN_WIDTH=11
# =================================================

# Color definitions
RESET='\033[0m'
GREEN='\033[38;5;82m'
YELLOW='\033[38;5;220m'
RED='\033[38;5;196m'
BLUE='\033[38;5;39m'
GRAY='\033[90m'
BOLD='\033[1m'

CURRENT_HOST=$(hostname -s)

# === Color block function ===
get_color_block() {
    local val=$1
    local char="■"
    local clean_val=${val//[^0-9]/} 

    if [ -z "$clean_val" ]; then
        echo -ne "${GRAY}${char}${RESET}"
        return
    fi
    if [ "$clean_val" -ge 80 ]; then
        echo -ne "${RED}${char}${RESET}"
    elif [ "$clean_val" -ge 40 ]; then
        echo -ne "${YELLOW}${char}${RESET}"
    else
        echo -ne "${GREEN}${char}${RESET}"
    fi
}

# ================= 1. Print global header =================
echo -e "${BOLD}=== Cluster GPU Monitor ===${RESET}  $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "Legend: ${GREEN}■${RESET} <40%   ${YELLOW}■${RESET} 40-80%   ${RED}■${RESET} >80%   ${GRAY}■${RESET} N/A"

# Dynamically calculate block column width
H_WIDTH=$((ALIGN_WIDTH * 2))

# Adjust header order: GPU UTIL -> VRAM USE -> CAP
# Added spaces (   ) between the two block groups
printf "${GRAY}%-10s %-12s %-${H_WIDTH}s   %-${H_WIDTH}s   %-8s${RESET}\n" \
    "NODE" "PART" "GPU UTIL" "VRAM USE" "CAP"

echo "----------------------------------------------------------------------------------------------------------------"

RAW_PARTITIONS=$(sinfo -h -o "%P" | sed 's/\*//')

for PARTITION in $RAW_PARTITIONS; do
    if [[ "$PARTITION" =~ $EXCLUDE_PARTS ]]; then continue; fi
    
    RAW_NODES=$(sinfo -p "$PARTITION" -h -o "%N")
    if [ "$RAW_NODES" == "n/a" ] || [ -z "$RAW_NODES" ]; then continue; fi

    mapfile -t NODES < <(scontrol show hostnames "$RAW_NODES")

    for node in "${NODES[@]}"; do
        NODE_STATE=$(sinfo -n "$node" -h -o "%t")
        if [[ "$NODE_STATE" == *"down"* ]] || [[ "$NODE_STATE" == *"drain"* ]]; then continue; fi

        CMD_NVIDIA="nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        
        if [[ "$node" == "$CURRENT_HOST" ]]; then
            remote_data=$(eval "$CMD_NVIDIA" 2>/dev/null)
        else
            remote_data=$(timeout "$TIMEOUT_SEC" srun --partition="$PARTITION" \
                           --nodelist="$node" --nodes=1 --ntasks=1 --oversubscribe --overlap --quiet \
                           $CMD_NVIDIA 2>/dev/null)
        fi

        if [ -z "$remote_data" ]; then
             printf "%-10s %-12s ${YELLOW}TIMEOUT/BUSY${RESET}\n" "$node" "${PARTITION:0:12}"
             continue
        fi

        # === Data processing ===
        util_blocks=""
        mem_blocks=""
        gpu_count=0
        vram_cap="-"
        
        while IFS=, read -r util mem_used mem_total; do
            util=$(echo $util | xargs)
            mem_used=$(echo $mem_used | xargs)
            mem_total=$(echo $mem_total | xargs)
            
            # Extract first GPU's capacity
            if [ $gpu_count -eq 0 ]; then
                clean_mem_total=${mem_total//[^0-9]/}
                if [ -n "$clean_mem_total" ] && [ "$clean_mem_total" -gt 0 ]; then
                    # Add 512MB for rounding, convert to GB
                    vram_cap=$(( (clean_mem_total + 512) / 1024 ))"GB"
                fi
            fi

            # Calculate percentage
            clean_mem_used=${mem_used//[^0-9]/}
            clean_mem_total=${mem_total//[^0-9]/}
            if [ -z "$clean_mem_total" ] || [ "$clean_mem_total" -eq 0 ] || [ -z "$clean_mem_used" ]; then 
                mem_pct="N/A"
            else 
                mem_pct=$(( (clean_mem_used * 100) / clean_mem_total ))
            fi
            
            # Build color blocks
            util_blocks+=$(get_color_block "$util")" "
            mem_blocks+=$(get_color_block "$mem_pct")" "
            
            ((gpu_count++))
        done <<< "$remote_data"

        # === Alignment padding ===
        pad_slots=$(( ALIGN_WIDTH - gpu_count ))
        if [ $pad_slots -lt 0 ]; then pad_slots=0; fi
        padding=$(printf '%*s' $((pad_slots * 2)) "")

        # Print single line
        # Order: Util -> Padding -> Mem -> Padding -> Cap
        # Added 3 spaces to match header spacing
        printf "%-10s ${BLUE}%-12s${RESET} %s%s   %s%s   ${BOLD}%-8s${RESET}\n" \
            "$node" "${PARTITION:0:12}" "$util_blocks" "$padding" "$mem_blocks" "$padding" "$vram_cap"

    done
done