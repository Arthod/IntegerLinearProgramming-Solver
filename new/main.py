class LinearProgram:
    def __init__(self):

        self.variables = []
        self.obj_function = 0
        self.constraints = []
        self.is_maximizing = False  # Default minimization

