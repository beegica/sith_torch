
import sys
from collections import deque
from random import randrange
from skimage.transform import resize
from ale_python_interface import ALEInterface
import numpy as np

class AleEnv(object):
    '''ALE wrapper for RL training
    game_over_conditions={'points':(-1, 1)}: dict that describes all desired game over conditions
    each key corresponds to a condition that is checked; the first condition met produces a game over
        points: int or tuple of integers
            int:
                if x < 0, game ends when score is <= x
                if x >= 0, game ends when score is >= x
            tuple:
                game ends if score <= x[0] or score >= x[1]
        lives: int that ends game when lives <= x
        frames: int that ends game when total number of frames >= x
        episodes: int that ends game when num of episodes >= x
            Use max_num_frames_per_episode to set max episode length


    '''
    # will include timing and hidden functionality in future iterations

    def __init__(
            self, rom_file, display_screen=False, sound=False,
            random_seed=0, game_over_conditions={'points':(-1, 1)},
            frame_skip=1, repeat_action_probability=0.25,
            max_num_frames_per_episode=0, min_action_set=False,
            screen_color='gray', fps=60, output_buffer_size=1,
            reduce_screen=False):

        # ALE instance and setup
        self.ale = ALEInterface()
        #TODO: check if rom file exists; will crash jupyter kernel otherwise
        self.ale.loadROM(str.encode(rom_file))

        self.ale.setBool(b'sound', sound)
        self.ale.setBool(b'display_screen', display_screen)

        if min_action_set:
            self.legal_actions = self.ale.getMinimalActionSet()
        else:
            self.legal_actions = self.ale.getLegalActionSet()

        self.ale.setInt(b'random_seed', random_seed)
        self.ale.setInt(b'frame_skip', frame_skip)
        self.frame_skip = frame_skip
        self.ale.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.ale.setInt(b'max_num_frames_per_episode', max_num_frames_per_episode)

        self.ale.loadROM(str.encode(rom_file))

        # environment variables
        self.total_points = 0
        self.game_over_conditions = game_over_conditions
        self.total_frames = self.ale.getFrameNumber()
        self.curr_episode = 1
        self.prev_ep_frame_num = -float("inf")

        self.reduce_screen = reduce_screen
        self.screen_color = screen_color
        if self.screen_color == 'gray' or self.screen_color == 'grey':
            self.game_screen = np.squeeze(self.ale.getScreenGrayscale())
            if self.reduce_screen:
                self.game_screen = resize(self.game_screen, output_shape=(110, 84))
                self.game_screen = self.game_screen[0+21:-1-4, :]
                self.game_screen[self.game_screen > 0.0] = 1
                #self.game_screen = self.game_screen[::2, ::2]
        elif self.screen_color == 'rgb' or self.screen_color == 'color':
            self.game_screen = self.ale.getScreenRGB()
            if self.reduce_screen:
                self.game_screen = resize(self.game_screen, output_shape=(110, 84, 3))
                self.game_screen = self.game_screen[0+21:-1-4, :, :]
                #self.game_screen = self.game_screen[::2, ::2]
        else:
            raise RuntimeError("ERROR: Invalid screen_color value")

        self.d_frame = (fps**-1) * self.frame_skip

        # set up output buffer
        self.output_buffer_size = output_buffer_size
        self.queue_size = self.output_buffer_size
        self.reset()

    def observe(self, flatten=False, expand_dim=False):

        if flatten is True:
            out = np.stack(self.output_queue[i] for i in range(self.output_buffer_size)).flatten()

            if expand_dim is True:
                return np.expand_dims(out, axis=0)
            else:
                return out
        else:
            out = np.stack(self.output_queue[i] for i in range(self.output_buffer_size))
            out = np.squeeze(out)

            if expand_dim is True:
                return np.expand_dims(out, axis=1)

            else:
                return out

    @property
    def width(self):
        return self.game_screen.shape[1]

    @property
    def height(self):
        return self.game_screen.shape[0]

    @property
    def game_over(self):
        return self._game_over()

    @property
    def actions(self):
        return self.legal_actions

    @property
    def lives(self):
        return self.ale.lives()

    def reset(self):
        self.ale.reset_game()
        self.output_queue = deque(np.zeros(shape=(self.queue_size -1,
                                                    self.height,
                                                    self.width)),
                            self.queue_size)
        self.output_queue.appendleft(self.game_screen)


    def act(self, action):
        reward = self.ale.act(self.legal_actions[action])
        if self.screen_color == 'gray' or self.screen_color == 'grey':
            self.game_screen = np.squeeze(self.ale.getScreenGrayscale())
            if self.reduce_screen:
                self.game_screen = resize(self.game_screen, output_shape=(110, 84))
                self.game_screen = self.game_screen[0+21:-1-4, :]
                self.game_screen[self.game_screen > 0.0] = 1
                #self.game_screen = self.game_screen[::2, ::2]
        elif self.screen_color == 'rgb' or self.screen_color == 'color':
            self.game_screen = self.ale.getScreenRGB()
            if self.reduce_screen:
                self.game_screen = resize(self.game_screen, output_shape=(110, 84, 3))
                self.game_screen = self.game_screen[0+21:-1-4, :, :]
                #self.game_screen = self.game_screen[::2, ::2, :]

        self.output_queue.pop()
        self.output_queue.appendleft(self.game_screen)

        self.total_points += reward
        self.total_frames += self.ale.getFrameNumber()
        if self.ale.getEpisodeFrameNumber() <= self.prev_ep_frame_num:
            self.curr_episode += 1
        self.prev_ep_frame_num = self.ale.getEpisodeFrameNumber()

        return reward, self.d_frame, self.game_over

    def _game_over(self):
        if self.ale.game_over():
            return True
        for cond in self.game_over_conditions:
            if cond == 'points':
                if isinstance(self.game_over_conditions[cond], int):
                    if self.total_points >= self.game_over_conditions[cond]:
                        return True
                elif isinstance(self.game_over_conditions[cond], tuple):
                    if (
                            self.total_points <= self.game_over_conditions[cond][0] or
                            self.total_points >= self.game_over_conditions[cond][1]):
                        return True
            elif cond == 'lives':
                if self.lives <= self.game_over_conditions[cond]:
                    return True
            elif cond == 'frames':
                if self.total_frames >= self.game_over_conditions[cond]:
                    return True
            elif cond == 'episodes':
                if self.curr_episode >= self.game_over_conditions[cond]:
                    return True
            else:
                raise RuntimeError("ERROR: Invalid game over condition")

        return False

if __name__ == "__main__":
    #ale = AleEnv("space_invaders.bin", min_action_set = True, display_screen=True)
    ale = AleEnv("breakout.a26", min_action_set = True, display_screen=True)
    print(ale.observe().shape)
    print(ale.width)
    print(ale.height)
    print(ale.actions)

    for i in range(2000):
        a = randrange(len(ale.actions))
        ale.act(a)
        print(ale.actions, ale.curr_episode, ale.lives, ale.total_points, a)
