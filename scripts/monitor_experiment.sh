#!/bin/bash
LOG_FILE="/tmp/experiment_100_mini.log"

while true; do
    clear
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  100-ITERATION GPT-4O-MINI EXPERIMENT MONITOR"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "[TIME] Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check current iteration
    CURRENT_ITER=$(grep "ITERATION" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "ITERATION [0-9]+" | grep -oE "[0-9]+")
    if [ -n "$CURRENT_ITER" ]; then
        PROGRESS=$((CURRENT_ITER * 100 / 100))
        echo "[INFO] Progress: Iteration $CURRENT_ITER/100 (${PROGRESS}%)"
        
        # Progress bar
        FILLED=$((PROGRESS / 2))
        BAR=$(printf '%*s' "$FILLED" | tr ' ' '█')
        EMPTY=$(printf '%*s' "$((50 - FILLED))" | tr ' ' '░')
        echo "   [$BAR$EMPTY] ${PROGRESS}%"
    else
        echo "[INFO] Progress: Initializing..."
    fi
    
    echo ""
    
    # Check for NEW BEST
    BEST_COUNT=$(grep -c "NEW BEST" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "[BEST] New Best Strategies Found: $BEST_COUNT"
    
    # Show latest best if exists
    if [ "$BEST_COUNT" -gt 0 ]; then
        LATEST_BEST=$(grep -A 2 "NEW BEST" "$LOG_FILE" | tail -3 | head -1 | sed 's/.*NEW BEST: //')
        LATEST_SCORE=$(grep -A 2 "NEW BEST" "$LOG_FILE" | tail -1 | grep -oE "Score: [0-9.]+" | grep -oE "[0-9.]+")
        if [ -n "$LATEST_SCORE" ]; then
            echo "   Latest: $LATEST_BEST"
            echo "   Score: $LATEST_SCORE/100"
            
            # Calculate gap to target
            GAP=$(echo "50.0 - $LATEST_SCORE" | bc 2>/dev/null)
            if [ -n "$GAP" ]; then
                echo "   Gap to Target: $GAP points"
            fi
        fi
    fi
    
    echo ""
    
    # Check for LOFO
    LOFO_COUNT=$(grep -c "\[LOFO\] Running ablation" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "[LOFO] LOFO Analyses Run: $LOFO_COUNT"
    
    # Check batch stats
    BATCH_COUNT=$(grep -c "\[BATCH\]" "$LOG_FILE" 2>/dev/null || echo "0")
    RETRY_COUNT=$(grep -c "\[RETRY\]" "$LOG_FILE" 2>/dev/null || echo "0")
    if [ "$BATCH_COUNT" -gt 0 ]; then
        RETRY_RATE=$((RETRY_COUNT * 100 / (BATCH_COUNT * 3) ))
        echo ""
        echo "[BATCH] Batch Generation:"
        echo "   Batch Calls: $BATCH_COUNT"
        echo "   Retries: $RETRY_COUNT (~${RETRY_RATE}% retry rate)"
    fi
    
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    
    # Check if complete
    if grep -q "REFINEMENT COMPLETE" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "[OK] EXPERIMENT COMPLETE!"
        echo ""
        echo "[SUCCESS] Generating visualizations..."
        break
    fi
    
    # Check if target reached
    if [ "$BEST_COUNT" -gt 0 ] && [ -n "$LATEST_SCORE" ]; then
        TARGET_REACHED=$(echo "$LATEST_SCORE >= 50.0" | bc 2>/dev/null)
        if [ "$TARGET_REACHED" = "1" ]; then
            echo ""
            echo "[TARGET] TARGET REACHED! Score >= 50.0"
            echo ""
        fi
    fi
    
    sleep 30
done
