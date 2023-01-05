import copy
import pygame
import random
import time
import sys


class Snake:
    def set_window_attr(self, xsize:int = 720, ysize:int = 480, 
                        colours:dict = {'black':pygame.Color(0, 0, 0), 'white':pygame.Color(255, 255, 255),
                                        'red':pygame.Color(255, 0, 0), 'green':pygame.Color(0, 255, 0),
                                        'blue':pygame.Color(0, 0, 255)}) -> None:
        """Set Snake game window attributes (window size, predefined object colours, text sizes and font)

        Args:
            xsize : xsize of playable window in pixels. Defaults to 720.
            ysize : xsize of playable window in pixels. Defaults to 480.
            colours : A dictionary of ('colour_name':pygame.Color) instances to be used globally.
        """
        self.x_window_size = xsize
        self.y_window_size = ysize
        self.colours = colours
        
        pygame.font.init()
        self.smallfont = pygame.font.SysFont('times new roman', 25)
        self.bigfont = pygame.font.SysFont('times new roman', 50)

    def set_snake_attr(self, snake_speed:int = 15, initial_snake_pos:list[list] = [[100, 50], [90, 50], [80, 50], [70, 50]],
                 initial_fruit_pos:tuple = None, initial_direction:str = 'RIGHT') -> None:
        """Set snake starting position, heading, and speed, and initial fruit position.

        Args:
            snake_speed (optional): Snake speed with respect to game refresh rate. Defaults to 15.
            initial_snake_pos (optional): A list of 2D lists indicating snake body start location. 
                                                      Defaults to [(100, 50), (90, 50), (80, 50), (70, 50)].
            initial_fruit_pos (optional): Initial fruit position as a 2D tuple. If set to None, a location is randomly chosen.
            initial_direction (optional): Starting snake initial heading. Defaults to 'RIGHT'.
        """

        self.speed = snake_speed
        self.snake_head = copy.deepcopy(initial_snake_pos)[0]
        self.snake_pos = copy.deepcopy(initial_snake_pos)
        
        if initial_fruit_pos is None:
            self.fruit_pos = (random.randrange(1, (self.x_window_size//10)) * 10, random.randrange(1, (self.y_window_size//10)) * 10)
        else:
            self.fruit_pos = initial_fruit_pos

        self.fruit_spawn = True
        
        self.direction = initial_direction
        self.change_to = initial_direction

    def set_game_attr(self, init_score:int = 0, score_increment:int = 10, game_caption:str = 'Snake game') -> None:
        """Set user initial score, fruit collection score increment, and game caption.

        Args:
            init_score (optional): The starting user score. Defaults to 0.
            score_increment (optional): The fruit catching score increment. Defaults to 10.
            game_caption (optional): The game window caption. Defaults to 'Snake game'.
        """

        self.score = init_score
        self.score_increment = score_increment
        self.fps = pygame.time.Clock()
        self.game_caption = game_caption

    def _show_score(self, color) -> None:
        """Show the user's score at the top left."""

        score_surface = self.smallfont.render('Score : ' + str(self.score), True, color) # obj to render
        score_rect = score_surface.get_rect()            # rectangular surface object
        self.game_window.blit(score_surface, score_rect) # blit obj to surface

    def _game_over_sequence(self, wait_time:int = 2) -> None:
        """Draw a 'GAME OVER' screen at the game's end and quit the application.
        
        Args:
            wait_time: The length of time of the 'GAME OVER' screen.
        """

        # Generate "game over" text and surface at a location
        font = self.bigfont
        game_over_surface = font.render('Your Score is : ' + str(self.score), True, self.colours['red'])
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.midtop = (self.x_window_size/2, self.y_window_size/4)
        
        # Blit
        self.game_window.blit(game_over_surface, game_over_rect)
        
        # End-screen wait + quit
        pygame.display.flip()
        time.sleep(wait_time)
        pygame.display.quit()
        pygame.quit()
        sys.exit()

    def _sense_keystroke(self) -> None:
        """Sense and store up/down/left/right keystrokes."""
        
        # handle key presses
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.change_to = 'UP'
                if event.key == pygame.K_DOWN:
                    self.change_to = 'DOWN'
                if event.key == pygame.K_LEFT:
                    self.change_to = 'LEFT'
                if event.key == pygame.K_RIGHT:
                    self.change_to = 'RIGHT'

        # restrict keypress results to only allow (-90,90) degree turns
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

    def _move_and_update_snake(self, walk_step_size: int = 10):
        """Move the snake in the corresponding prior keystroke direction `walk_step_size` units.

        Args:
            walk_step_size: The snake's frame-by-frame step-size with window pixels as the step-size unit. Defaults to 10.
        """

        # Find snake's new start position
        if self.direction == 'UP':
            self.snake_head[1] -= walk_step_size
        if self.direction == 'DOWN':
            self.snake_head[1] += walk_step_size
        if self.direction == 'LEFT':
            self.snake_head[0] -= walk_step_size
        if self.direction == 'RIGHT':
            self.snake_head[0] += walk_step_size

        # Add the new position to the snake's body
        self.snake_pos.insert(0, list(self.snake_head))

        # Maintain or remove the snake's tail given fruit collection
        if self.snake_head[0] == self.fruit_pos[0] and self.snake_head[1] == self.fruit_pos[1]:
            self.score += self.score_increment
            self.fruit_spawn = False
        else:
            self.snake_pos.pop()

    def _update_fruit(self):
        """Spawn in a new fruit if the old one has been taken."""

        if self.fruit_spawn is False:
            self.fruit_pos = (random.randrange(1, (self.x_window_size//10)) * 10,
                              random.randrange(1, (self.y_window_size//10)) * 10)
            self.fruit_spawn = True
    
    def _redraw_board(self):
        """Redraw the pygame window using the new (or old) snake and fruit locations."""
        
        # make the board black
        self.game_window.fill(self.colours['black'])
        
        # draw a green square for each snake pos., and a white square for a fruit
        for pos in self.snake_pos:
            pygame.draw.rect(self.game_window, self.colours['green'],
                             pygame.Rect(pos[0], pos[1], 10, 10))

        pygame.draw.rect(self.game_window, self.colours['white'],
                         pygame.Rect(self.fruit_pos[0], self.fruit_pos[1], 10, 10))

    def _test_game_over(self):
        """Test snake collisions."""
        # Snake hit walls
        if self.snake_head[0] < 0 or self.snake_head[0] > self.x_window_size-10:
            self._game_over_sequence()
        if self.snake_head[1] < 0 or self.snake_head[1] > self.y_window_size-10:
            self._game_over_sequence()

        # Snake collided into itself
        for block in self.snake_pos[1:]:
            if self.snake_head[0] == block[0] and self.snake_head[1] == block[1]:
                self._game_over_sequence()

    def play(self):
        """Play the snake game according to prior initialisations."""
        
        # initialise game instance, game window, and caption
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption(self.game_caption)
        self.game_window = pygame.display.set_mode((self.x_window_size, self.y_window_size))

        while(1):
            # sense, move, update board, and test for game over
            self._sense_keystroke()
            self._move_and_update_snake()
            self._update_fruit()
            self._redraw_board()
            self._show_score(self.colours['white'])
            self._test_game_over()

            # Refresh screen
            pygame.display.update()

            # Pause according to FPS requested
            self.fps.tick(self.speed)
        
        pygame.display.quit()
        pygame.quit()
        sys.exit()
