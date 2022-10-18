for ind in $(eval echo "{$2..$3}")
do
    for i in {0..4}
    do
        
    
        python3 main2tasks.py --data mnist --mod rot --seed $i --gpu $1 --inc 90 --tasks 4 --print file --index $ind --model resnet --train-until epoch --lr 0.1 --epochs 1 --strat naive --strat replay --num-examples 2000
    done 
done