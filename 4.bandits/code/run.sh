#!/bin/sh

echo "start recommending..."
START=$(date +%s)
python evaluator.py ../data/articles.txt ../data/log.txt
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
