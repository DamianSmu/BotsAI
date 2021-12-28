from NN import NN


class TestBot:
    def __init__(self, game, model_path, bot, is_random, nn):
        self.game = game
        self.bot = bot
        self.is_random = is_random
        if model_path is not None:
            self.model = nn.load_model(model_path)
        else:
            self.model = None

    def play(self):
        self.bot.set_state(self.game.get_objects(), map_size=20)
        start_objects = self.bot.units + self.bot.settlements
        for map_object in start_objects:
            global_state, local_state = self.bot.get_padded_state(map_object)
            action = self.bot.predict_action(self.model, map_object, self.is_random, global_state, local_state)
            if action[2] in [4, 5, 6, 7]:
                action[2] = 10
            self.game.post_actions_and_take_turn(action, self.bot)
        self.game.end_turn(self.bot)

