
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f'{self.x}_{self.y}'
    def __repr__(self):
        return f'{self.x}_{self.y}'
    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        else:
            return False
    def __hash__(self):
        return hash(str(self.x) + '_' + str(self.y))