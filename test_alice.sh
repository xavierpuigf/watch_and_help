# cd interface
# python main_bob.py --task setup_table --num-per-apartment 1 --mode full --use-editor
# cd ..

# cd interface
# python main_bob.py --task prepare_food --num-per-apartment 1 --mode full --use-editor
# cd ..


# cd interface
# python main_bob.py --task put_fridge --num-per-apartment 1 --mode full --use-editor
# cd ..


python test_alice.py --max-episode-length 250 --num-per-apartment 300 --mode check_neurips

# cd interface
# python main_bob.py --task read_book --num-per-apartment 1 --mode full --use-editor
# cd ..

