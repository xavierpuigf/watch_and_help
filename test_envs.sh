# cd init_reward
# python vh_init.py --task put_dishwasher --num-per-apartment 2
# cd ..
# cd interface
# python main_demo.py --task put_dishwasher --num-per-apartment 2
# cd ..

# cd init_reward
# python vh_init.py --task prepare_food --num-per-apartment 2
# cd ..
# cd interface
# python main_demo.py --task prepare_food --num-per-apartment 2
# cd ..

# cd init_reward
# python vh_init.py --task put_fridge --num-per-apartment 2
# cd ..
# cd interface
# python main_demo.py --task put_fridge --num-per-apartment 2
# cd ..

cd init_reward
python vh_init.py --task setup_table --num-per-apartment 2
cd ..
cd interface
python main_demo.py --task setup_table --num-per-apartment 2 --recording




