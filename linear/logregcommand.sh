for i in {0..5}
do
    python3 main.py $i
    wait -n $!
done
