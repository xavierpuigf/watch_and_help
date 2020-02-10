class Reward:
    def __init__(self, goal):
        self.goal = goal

    def setup_table(state):
        if (state_num_people >= goal['num_people']) &
            (state_num_plates >= goal['num_plates']) &
            (state_num_glasses >= goal['num_glasses']) &
            (state_num_wine >= goal['num_wine']) &
            (state_num_forks >= goal['num_forks']):
            return 1
        else:
            return 0
        

    def read_book(state):
        if (state_num_book >= goal['num_book']):
            return 1
        else:
            return 0

    def clean_table(state):
        if (state_num_food >= goal['num_food']) &
            (state_num_plate >= goal['num_plate']) &
            (state_num_glasses >= goal['num_glasses']) &
            (state_num_wine >= goal['num_wine']) &
            (state_num_forks >= goal['num_forks']):
            return 1
        else:
            return 0

        
    def put_diswasher(state):
        if (state_num_plate >= goal['num_plate']) &
            (state_num_glasses >= goal['num_glasses']) &
            (state_num_forks >= goal['num_forks']):
            return 1
        else:
            return 0


    def unload_diswasher(state):
        if (state_num_plate >= goal['num_plate']) &
            (state_num_glasses >= goal['num_glasses']) &
            (state_num_forks >= goal['num_forks']):
            return 1
        else:
            return 0

    def put_fridge(state):
        if (state_num_food >= goal['num_food']):
            return 1
        else:
            return 0

    def prepare_food(state):
        state_num_food >= goal['num_food']
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

    def watch_tv(state):
        goal['tv'] = 'off'





    # goals = [
    #     (setup_table, ),
    #     (read_book, ),
    #     (clean_table, ),
    #     (put_diswasher, ),
    #     (unload_diswasher, ),
    #     (put_fridge, ),
    #     (prepare_food, ),
    #     (watch_tv, ),

    #     (setup_table, prepare_food),
    #     (setup_table, read_book),
    #     (setup_table, watch_tv),
    #     (setup_table, put_fridge),
    #     (setup_table, put_fridge),
    #     (setup_table, put_diswasher),
    # ]


