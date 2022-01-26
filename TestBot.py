class TestBot:
    def __init__(self, game, model_path, bot, is_random, nn, forbidden_actions=None):
        self.game = game
        self.bot = bot
        self.is_random = is_random
        self.forbidden_actions = forbidden_actions
        if model_path is not None:
            self.model = nn.load_model(model_path)
        else:
            self.model = None

    def play(self):
        self.bot.set_state(self.game.get_objects(), map_size)
        start_objects = self.bot.units + self.bot.settlements
        for map_object in start_objects:
            global_state, local_state = self.bot.get_padded_state(map_object)
            action = self.bot.predict_action(self.model, map_object, self.is_random, global_state, local_state)
            if self.forbidden_actions is not None and action[2] in self.forbidden_actions:
                action[2] = 10
            self.game.post_actions_and_take_turn(action, self.bot)
        self.game.end_turn(self.bot)
        self.bot.set_state(self.game.get_objects(), map_size)
