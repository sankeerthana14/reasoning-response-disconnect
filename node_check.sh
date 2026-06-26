#!/bin/bash

# ================= Config =================
TIMEOUT="15" 
# ==========================================

INPUT=$1

if [ -z "$INPUT" ]; then
    echo "Usage: nodecheck <node_number>"
    exit 1
fi

if [[ "$INPUT" =~ ^[0-9]+$ ]]; then
    NODE="node$INPUT"
else
    NODE="$INPUT"
fi

# 1. Auto-detect partition
DETECTED_PARTITION=$(scontrol show node "$NODE" 2>/dev/null | grep "Partitions=" | cut -d'=' -f2 | awk -F',' '{print $1}' | xargs)
if [ -z "$DETECTED_PARTITION" ]; then
    DETECTED_PARTITION=$(sinfo -n "$NODE" -h -o "%P" 2>/dev/null | head -n 1 | tr -d '*' | cut -d',' -f1 | xargs)
fi

if [ -z "$DETECTED_PARTITION" ] || [ "$DETECTED_PARTITION" == "n/a" ]; then
    echo -e "\033[31mError: Node '$NODE' not found or invalid.\033[0m"
    exit 1
fi

# 2. Define remote script
REMOTE_SCRIPT='
    RESET=$(printf "\033[0m")
    GREEN=$(printf "\033[32m")
    YELLOW=$(printf "\033[33m")
    RED=$(printf "\033[31m")
    CYAN=$(printf "\033[36m")
    GRAY=$(printf "\033[90m")
    BOLD=$(printf "\033[1m")

    echo -e "${BOLD}=== $HOSTNAME: Check Report ===${RESET}  $(date "+%Y-%m-%d %H:%M:%S")"
    
    # Print header: adjust column widths
    # ID(4) MODEL(20) UTIL(10) VRAM(16) USERS(Auto)
    printf "${GRAY}%-4s %-20s %-10s %-16s %s${RESET}\n" "ID" "MODEL" "UTIL(%)" "VRAM(Use/Cap)" "USERS"
    echo "------------------------------------------------------------------------------------------------"
    
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r idx name util mem_used mem_total; do
        
        idx=$(echo $idx | xargs)
        name=$(echo $name | xargs)
        util=$(echo $util | xargs)
        mem_used=$(echo $mem_used | xargs)
        mem_total=$(echo $mem_total | xargs)
        
        # --- 1. Model full name ---
        full_name=$(echo "$name" | sed "s/NVIDIA //g")
        
        # --- 2. GPU utilization color ---
        ucolor=$GREEN
        if [ "$util" -ge 80 ]; then ucolor=$RED; elif [ "$util" -ge 40 ]; then ucolor=$YELLOW; fi
        
        # --- 3. VRAM calculation ---
        if [ -z "$mem_total" ] || [ "$mem_total" -eq 0 ]; then
            mem_pct=0
            mem_str="N/A"
            vcolor=$GREEN
        else
            # Percentage
            mem_pct=$(( (mem_used * 100) / mem_total ))
            
            # Capacity in GB (rounded: +512)
            gb_used=$(( mem_used / 1024 ))
            gb_total=$(( (mem_total + 512) / 1024 ))
            
            # Format: 44%  35/80GB
            mem_str="${mem_pct}%  ${gb_used}/${gb_total}GB"
            
            # VRAM color
            vcolor=$GREEN
            if [ "$mem_pct" -ge 80 ]; then vcolor=$RED; elif [ "$mem_pct" -ge 40 ]; then vcolor=$YELLOW; fi
        fi
        
        # --- 4. User info (with VRAM usage) ---
        # Get PID and Used Memory (MB)
        # Result looks like: "4123, 1024"
        app_rows=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i $idx)
        
        user_info_str=""
        if [ -z "$app_rows" ]; then
            user_info_str="${GREEN}(Idle)${RESET}"
        else
            first=1
            # Read application info line by line
            while IFS=, read -r pid app_mem_mb; do
                pid=$(echo $pid | xargs)
                app_mem_mb=$(echo $app_mem_mb | xargs)
                
                if [ -n "$pid" ]; then
                    user=$(ps -o user= -p $pid 2>/dev/null)
                    cmd=$(ps -o comm= -p $pid 2>/dev/null)
                    
                    # Process VRAM MB -> GB (rounded)
                    # If app_mem_mb is empty, default to 0
                    if [ -z "$app_mem_mb" ]; then app_mem_mb=0; fi
                    app_mem_gb=$(( (app_mem_mb + 512) / 1024 ))
                    
                    if [ -n "$user" ]; then
                        if [ $first -eq 0 ]; then user_info_str+=", "; fi
                        
                        # Format: user(cmd:12GB)
                        # Username in cyan, other info in white
                        user_info_str+="${CYAN}${user}${RESET}(${cmd}:${app_mem_gb}GB)"
                        first=0
                    fi
                fi
            done <<< "$app_rows"
        fi

        # --- 5. Print single line ---
        printf "%-4s %-20s %s%-10s%s %s%-16s%s %s\n" \
            "$idx" \
            "${full_name:0:20}" \
            "$ucolor" "${util}%" "$RESET" \
            "$vcolor" "${mem_str}" "$RESET" \
            "$user_info_str"
    done
'

# 3. Submit task
timeout "$TIMEOUT" srun --partition="$DETECTED_PARTITION" \
     --nodelist="$NODE" \
     --nodes=1 \
     --ntasks=1 \
     --oversubscribe \
     --overlap \
     --quiet \
     bash -c "$REMOTE_SCRIPT"

# Error handling
RET_CODE=$?
if [ $RET_CODE -ne 0 ]; then
    if [ $RET_CODE -eq 124 ]; then
        echo -e "\033[33mError: Timeout connecting to $NODE.\033[0m"
    else
        echo -e "\033[31mError: Connection failed. Check node status.\033[0m"
    fi
fi