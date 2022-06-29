for i in range(0, 120, 15):
    print(f"python3 main.py --mod noise --gpu {i//15} --inc 0.03 --tasks 5 --start {i} --end {i+15} --samples 5 --seed {i//15} --print file &")
