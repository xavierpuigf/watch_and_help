def setup_table(initial):
    init_num_people >= initial['num_people']
    init_num_plates >= initial['num_plates']
    init_num_glasses >= initial['num_glasses']
    init_num_wine >= initial['num_wine']
    init_num_forks >= initial['num_forks']

    

def read_book(initial):
    init_num_book >= initial['num_book']
    
    ## goal
    # pos = initial['where_to_read']
    # num_obj_inplace = initial['num_obj_inplace']


def clean_table(initial):
    init_num_food >= initial['num_food']
    init_num_plate >= initial['num_plate']
    init_num_glasses >= initial['num_glasses']
    init_num_wine >= initial['num_wine']
    init_num_forks >= initial['num_forks']

    
def put_diswasher(initial):
    init_num_plate >= initial['num_plate']
    init_num_glasses >= initial['num_glasses']
    init_num_forks >= initial['num_forks']

def unload_diswasher(initial):
    init_num_plate >= initial['num_plate']
    init_num_glasses >= initial['num_glasses']
    init_num_forks >= initial['num_forks']

def put_fridge(initial):
    init_num_food >= initial['num_food']

def prepare_food(initial):
    init_num_food >= initial['num_food']
    FOOD_APPLE
    FOOD_CEREAL
    FOOD_BANANA
    FOOD_BREAD
    FOOD_CARROT
    FOOD_CHICKEN
    FOOD_DESSERT
    FOOD_FISH
    FOOD_HAMBURGER
    FOOD_LEMON
    FOOD_LIME
    FOOD_OATMEAL
    FOOD_POTATO
    FOOD_SALT
    FOOD_SNACK
    FOOD_SUGAR
    FOOD_TURKEY

def watch_tv(initial):
    initial['tv'] = 'off'


goals = [
    (setup_table, ),
    (read_book, ),
    (clean_table, ),
    (put_diswasher, ),
    (unload_diswasher, ),
    (put_fridge, ),
    (prepare_food, ),
    (watch_tv, ),

    (setup_table, prepare_food),
    (setup_table, read_book),
    (setup_table, watch_tv),
    (setup_table, put_fridge),
    (setup_table, put_fridge),
    (setup_table, put_diswasher),
]





