

sort -t $'\t' -k 1,1 -k 4,4 ../download/ECommAI_ubp_round2_train > ../download/ECommAI_ubp_round2_train_sort

python data_gen/gen_behavior.py
python data_gen/gen_neg_sample.py

