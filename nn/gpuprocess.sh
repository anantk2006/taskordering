for ind in $(eval echo "{$2..$3}")
do
    for i in {0..3}
    do
        
    
        python3 main2tasks.py --data mnist --mod rot --seed $i --gpu $1 --inc 30 --tasks 3 --print file --index $((ind*720)) --model resnet --train-until loss --lr 0.1 --loss-thres 0.05 --strat naive
    done 
done