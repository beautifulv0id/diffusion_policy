for file in *_mask_v2.sh; 
do mv "$file" "${file/_mask_v2/}"; 
done
