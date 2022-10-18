for i in {0..9}
do
    python3 logregv7.py $i
    wait -n $!
done
