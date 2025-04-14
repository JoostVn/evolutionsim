import numpy as np

class Legend:
    
    def __init__(self, plot, location, width, border):

        # Parent plot and layout attributes
        self.plot = plot
        self.border = border

        # Spacing and location constants
        self.location = location
        self.width = width
        self.x_spacing = 10
        self.y_spacing = 16
        self.label_yoffset = -5
        self.handle_dim = (6,6)

        # Variable dimensions (NW/SE corners + line positioning)
        self.nw, self.se, self.handle_pos, self.label_pos = None, None, None, None
        self.update_dimensions()

    def update_dimensions(self):
        """
        Computes the NW and SE corners of the legend box, as well as the
        lable and handle positioning. Call this function after changing
        plot domain/dim/pos or adding graph elements.
        """

        # North west corner
        margin = 4
        if self.location == 'outer right':
            x = self.plot.pos[0] + self.plot.dim[0] + margin
            y = self.plot.pos[1]
        elif self.location == 'upper right':
            x = self.plot.pos[0] + self.plot.dim[0] - margin - self.width
            y = self.plot.pos[1] + margin
        elif self.location == 'upper left':
            x = self.plot.pos[0] + margin
            y = self.plot.pos[1] + margin
        self.nw = (x,y)

        # South east corner
        height = self.y_spacing * len(self.plot.elements)
        self.se = np.add(self.nw, (self.width, height))

        # Handle and label positions
        handle_base = np.add(self.nw, self.handle_dim)
        label_base = np.add(handle_base, (self.x_spacing, self.label_yoffset))
        self.handle_pos, self.label_pos = [], []
        for i, element in enumerate(self.plot.elements.values()):
            self.handle_pos.append(np.add(handle_base, (0, i  * self.y_spacing)))
            self.label_pos.append(np.add(label_base, (0, i * self.y_spacing)))

    def draw(self, screen):
        """
        Draws the legend box, handles and labels.
        """
        self.plot.pdraw.pgc_rect(
            screen, self.plot.pdraw.color_bg, self.nw, self.se, self.border)
        legend_lines = zip(self.handle_pos, self.label_pos, self.plot.elements.values())
        for handle, label, element in legend_lines:
            handle_rect_dim = (handle, np.add(handle, self.handle_dim))
            self.plot.pdraw.pgc_rect(screen, element.color, *handle_rect_dim)
            self.plot.pdraw.pgc_text(screen, element.label, label, self.plot.pdraw.font_m)

       
