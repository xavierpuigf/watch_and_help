
# cd interface
# python main_demo.py --task put_dishwasher --num-per-apartment 1 #--recording
# cd ..

# cd init_reward
# python vh_init.py --task prepare_food --num-per-apartment 1
# cd ..
# cd interface
# python main_demo.py --task prepare_food --num-per-apartment 1 #--recording
# cd ..

#cd init_reward
#python vh_init.py --task put_fridge --num-per-apartment 50
#cd ..
#cd interface
#python main_demo.py --task put_fridge --num-per-apartment 50 --recording
#cd ..
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

#cd init_reward
#python vh_init.py --task setup_table --num-per-apartment 50
#cd ..
#cd interface
#python main_demo.py --task setup_table --num-per-apartment 50 --recording

#cd init_reward
#python vh_init.py --task read_book --num-per-apartment 50
#cd ..

#cd interface
#python main_demo.py --task prepare_food --num-per-apartment 50 --mode full --port 8080 --display "3"  --recording
#cd ..

#cd interface
#python main_demo.py --task setup_table --num-per-apartment 50 --mode full --recording --display "3" --port 8090 --recording
#cd ..

#cd init_reward
#python vh_init.py --task prepare_food --num-per-apartment 50 --mode full --port 8210
#cd ..

#cd init_reward
#python vh_init.py --task put_fridge --num-per-apartment 50 --mode full --port 8210 --display "3"
#cd ..



#cd interface
#python main_demo.py --task setup_table --num-per-apartment 50 --recording --mode full --port 8090 --display 2 
#cd ..


cd interface
python main_demo.py --task setup_table --num-per-apartment 50 --recording --mode full --port 8088 --display 5
python main_demo.py --task read_book --num-per-apartment 50 --recording --mode full --port 8088 --display 5
