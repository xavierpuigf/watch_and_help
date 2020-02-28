# cd init_reward
# python vh_init.py --task put_dishwasher --num-per-apartment 1
# cd ..
# cd interface
# python main_demo.py --task put_dishwasher --num-per-apartment 1 #--recording
# cd ..

# cd init_reward
# python vh_init.py --task prepare_food --num-per-apartment 1
# cd ..
# cd interface
# python main_demo.py --task prepare_food --num-per-apartment 1 #--recording
# cd ..

# cd init_reward
# python vh_init.py --task put_fridge --num-per-apartment 1
# cd ..
# cd interface
# python main_demo.py --task put_fridge --num-per-apartment 1 #--recording
# cd ..

# cd init_reward
# python vh_init.py --task setup_table --num-per-apartment 1
# cd ..
# cd interface
# python main_demo.py --task setup_table --num-per-apartment 1 #--recording

cd init_reward
python vh_init.py --task read_book --num-per-apartment 1
cd ..
cd interface
python main_demo.py --task read_book --num-per-apartment 1 #--recording
cd ..



