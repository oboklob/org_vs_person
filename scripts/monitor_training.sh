#!/bin/bash
# Monitor training progress and alert when complete

PROCESS_ID=$1

if [ -z "$PROCESS_ID" ]; then
    echo "Usage: $0 <process_id>"
    exit 1
fi

echo "Monitoring process $PROCESS_ID..."
echo "Started at: $(date)"
echo "Press Ctrl+C to stop monitoring"
echo ""

while kill -0 $PROCESS_ID 2>/dev/null; do
    # Get memory usage
    mem=$(ps -p $PROCESS_ID -o rss= 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024}')
    cpu=$(ps -p $PROCESS_ID -o %cpu= 2>/dev/null)
    elapsed=$(ps -p $PROCESS_ID -o etime= 2>/dev/null | xargs)
    
    echo "$(date +%H:%M:%S) | Elapsed: $elapsed | CPU: ${cpu}% | Memory: $mem"
    sleep 30
done

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "==========================================" 
echo ""
echo "Analyzing results..."
python scripts/analyze_features.py
