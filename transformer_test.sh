# Testing transformer model on the threesym case with {x0x1x2=1} is the unseen domain.
# f(x) = x0x1 - 1.25x1x2 + 1.5 x0x2
# f'(x) = x2 -1.25x0 + 1.5x1 is then the MD interpolator
python3 main.py -seed 1 -task threesym -model transformer -lr 0.00002 -epochs 10 -batch-size 256 -opt adam

# Testing the transformer model on the 2parity case with {(x0,x1) = (-1,-1)} is the unseen domain.
# f(x) = x0x1
# f'(x) = x0 + x1 - 1 is then the MD interpolator.
#python3 main.py -seed 1 -task 2parity -model transformer -lr 0.00001 -epochs 10 -batch-size 256 -opt adam

# Testing the transformer model on the cyclic3 case with {(x0,x1,x2) = (-1,-1,-1)} is the unseen domain.
# f(x) = x0x1x2 + x1x2x3 + ... + x13x14x0 + x14x0x1
# f'(x) = (x0x1 + x1x2 + x0x2 - x0 - x1 - x2 +1) + x1x2x3 + ... + x14x0x1
#python3 main.py -seed 1 -task cyclic3dim15 -model transformer -lr 0.00002 -epochs 10 -batch-size 256 -opt adam

# Testing the transformer model on the maj(x0x1x2) case.
#python3 main.py -seed 1 -task maj3dim40freeze2 -model transformer -lr 0.00001 -epochs 10 -batch-size 256 -opt adam
