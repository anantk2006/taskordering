for i in range(0, 120, 15):
    print(f"python3 rotated_cifar.py --data rot --gpu {i//15} --inc 30 --tasks 5 --start {i} --end {i+15} --samples 5 &")
