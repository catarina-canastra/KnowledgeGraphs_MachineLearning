#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <algorithm> <relationship_id> <top_k>"
    exit 1
fi

algorithm="$1"
relationship_id="$2"
top_k="$3"

input_file="unique_entities_test_set.txt"

if [ ! -e "$input_file" ]; then
    echo "Input file not found: $input_file"
    exit 1
fi

entity_counter=0  # Initialize the counter

while read -r entity_type entity_value
do
    entity_counter=$((entity_counter + 1))
    echo "Processing entity #$entity_counter: Type=$entity_type, Value=$entity_value"
    time python3.6 New_Full_Test_Model.py --algorithm "$algorithm" --entity_type "$entity_type" --entity "$entity_value" --relation_id "$relationship_id" --top_k "$top_k"
done < "$input_file"

echo "Total predictions made: $entity_counter"
