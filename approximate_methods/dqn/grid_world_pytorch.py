from grid_world import GridWorld
game = GridWorld(size=4, mode='static')

print(game.display())
game.makeMove('d')
game.makeMove('d')
game.makeMove('l')
print(game.display())
print(game.reward())