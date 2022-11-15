for i in {0..39}
do
    python3 main.py $i
    wait -n $!
done
