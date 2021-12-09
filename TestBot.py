from NN import NN


class TestBot:
    def __init__(self, game, model_path, bot, is_random):
        self.game = game
        self.bot = bot
        self.is_random = is_random
        if model_path is not None:
            nn = NN()
            self.model = nn.load_model(model_path)
        else:
            self.model = None

    def play(self):
        self.bot.set_state(self.game.get_objects(), map_size=20)
        start_objects = self.bot.map_objects
        for map_object in start_objects:
            global_state, local_state = self.bot.get_padded_state(map_object)
            action = self.bot.predict_action(self.model, map_object, self.is_random, global_state, local_state)
            self.game.post_actions_and_take_turn(action, self.bot)
        self.game.end_turn(self.bot)

