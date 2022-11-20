for i in {2..14}
do
    python3 main.py $i
    wait -n $!
done
