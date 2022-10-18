for ind in {0..7}
do 
    bash gpuprocess.sh $ind $((ind*63)) $((ind*63+62)) &
done