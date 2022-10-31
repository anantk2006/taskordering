for i in {0..9}
do
    python3 logreg.py $i
    wait -n $!
done
