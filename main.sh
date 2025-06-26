#!/bin/bash


# Loop over alpha values
for lr in 0.01 0.001 0.0001
do
    # Loop over learning rate (eta) values
    for batch_size in 20 32 64 128
    do   
        for hidden_units in 5 12 30 60 120 
        do
            for embedding in 0 1
            do
                for noise_scale in 0 0.0001 0.001
                do
                        # Run the Python script with the current values of alpha and eta
                    python /home/users/scro4690/Documents/GenInv/Gassmann/src/example/full/FMPE.py --lr $lr --batch_size $batch_size --hidden_units $hidden_units --embedding $embedding --noise_scale $noise_scale
                done
            done
        done
    done
done
