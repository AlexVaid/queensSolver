from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.core.image import Image as CoreImage
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.animation import Animation
from kivy.properties import BooleanProperty, ListProperty
import requests
import base64
import io
from plyer import filechooser

# -----------------------------
# Utility Function
# -----------------------------

def adjust_color(color, amount):
    """
    Adjusts the brightness of a color.
    :param color: A list of [r, g, b, a]
    :param amount: Float between -1 and 1. Positive to lighten, negative to darken.
    :return: Adjusted color.
    """
    return [max(min(c + amount, 1), 0) for c in color[:3]] + [color[3]]

# -----------------------------
# Hover Behaviors
# -----------------------------

class HoverBehavior:
    """
    Mixin class that adds hover behavior to widgets.
    """
    hovered = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouse_pos)
        self.register_event_type('on_enter')
        self.register_event_type('on_leave')

    def on_mouse_pos(self, window, pos):
        if not self.get_root_window():
            return  # Do nothing if widget is not displayed yet

        inside = self.collide_point(*self.to_widget(*pos))
        if self.hovered == inside:
            return

        self.hovered = inside
        if inside:
            self.dispatch('on_enter')
        else:
            self.dispatch('on_leave')

    def on_enter(self):
        pass

    def on_leave(self):
        pass

class HoverBackgroundColorBehavior(HoverBehavior):
    """
    Mixin class that adds hover background color change behavior to widgets.
    """
    original_color = ListProperty([0.7, 0.7, 0.7, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = self.original_color
        self.hover_amount = 0.1  # Amount to lighten the color on hover
        self.hover_color = adjust_color(self.original_color, self.hover_amount)

    def on_enter(self, *args):
        Animation.cancel_all(self, 'background_color')
        Animation(background_color=self.hover_color, duration=0.1).start(self)

    def on_leave(self, *args):
        Animation.cancel_all(self, 'background_color')
        Animation(background_color=self.original_color, duration=0.1).start(self)

# -----------------------------
# Custom Widgets
# -----------------------------

class HoverButton(HoverBackgroundColorBehavior, Button):
    """
    Button widget with hover behavior.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.font_size = kwargs.get('font_size', '20sp')
        self.pressed_amount = -0.1  # Amount to darken the color when pressed
        self.pressed_color = adjust_color(self.original_color, self.pressed_amount)

    def on_press(self):
        Animation.cancel_all(self, 'background_color')
        Animation(background_color=self.pressed_color, duration=0.1).start(self)

    def on_release(self):
        Animation.cancel_all(self, 'background_color')
        Animation(background_color=self.hover_color, duration=0.1).start(self)

class HoverTextInput(HoverBackgroundColorBehavior, TextInput):
    """
    TextInput widget with hover behavior and custom styling.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('foreground_color', [0, 0, 0, 1])
        kwargs.setdefault('cursor_color', [0, 0, 0, 0])  # Hide cursor
        super().__init__(**kwargs)
        self.font_size = kwargs.get('font_size', '15sp')
        self.bind(focus=self.on_focus)

    def on_focus(self, instance, value):
        # Prevent focus
        if value:
            self.focus = False

    def insert_text(self, substring, from_undo=False):
        # Disable manual text input
        pass

# -----------------------------
# Custom Image with White Background
# -----------------------------

class WhiteBackgroundImage(KivyImage):
    """
    Image widget with a white background.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(1, 1, 1, 1)  # White color
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

# -----------------------------
# Main Application Widget
# -----------------------------

class MainWidget(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.font_size = '20sp'  # Increased font size for all widgets

        # Create UI components
        self.selected_image = None
        self.create_board_size_input()
        self.create_action_buttons()
        self.create_result_image()

    def create_board_size_input(self):
        """Creates the input field for board size with increment and decrement buttons."""
        # Label for board size
        self.n_label = Label(
            text='Enter board size (N):',
            font_size=self.font_size,
            size_hint=(0.3, 0.05),
            pos_hint={'x': 0.03, 'y': 0.9}
        )
        self.add_widget(self.n_label)

        # Layout for input and buttons
        input_layout = BoxLayout(
            orientation='horizontal',
            size_hint=(0.3, 0.09),
            pos_hint={'x': 0.03, 'y': 0.8}
        )

        button_kwargs = {
            'font_size': self.font_size,
            'background_color': [0.7, 0.7, 0.7, 1],
            'size_hint': (0.3, 1)
        }

        # Decrement button
        self.decrement_button = HoverButton(
            text='-',
            **button_kwargs
        )
        self.decrement_button.bind(on_press=self.decrement_value)
        input_layout.add_widget(self.decrement_button)

        # Input field with hover behavior
        self.n_input = HoverTextInput(
            text='7',
            multiline=False,
            halign='center',
            readonly=True,
            font_size='15sp',
            background_color=[0.7, 0.7, 0.7, 1],
            size_hint=(0.4, 1),
            pos_hint={'x': 0, 'y': 0.01}  # Align with increment buttons
        )
        self.n_input.bind(size=self.update_text_size)
        self.n_input.padding = [0, 0]
        input_layout.add_widget(self.n_input)

        # Increment button
        self.increment_button = HoverButton(
            text='+',
            **button_kwargs
        )
        self.increment_button.bind(on_press=self.increment_value)
        input_layout.add_widget(self.increment_button)

        self.add_widget(input_layout)

    def create_action_buttons(self):
        """Creates the buttons for selecting an image and solving the puzzle."""
        button_kwargs = {
            'font_size': self.font_size,
            'background_color': [0.7, 0.7, 0.7, 1],
            'size_hint': (0.3, 0.15)
        }

        # Button to select image
        self.image_button = HoverButton(
            text='Select an image',
            pos_hint={'x': 0.35, 'y': 0.8},
            **button_kwargs
        )
        self.image_button.bind(on_press=self.select_image)
        self.add_widget(self.image_button)

        # Button to solve the puzzle
        self.solve_button = HoverButton(
            text='Solve puzzle',
            pos_hint={'x': 0.67, 'y': 0.80},
            **button_kwargs
        )
        self.solve_button.bind(on_press=self.solve_puzzle)
        self.add_widget(self.solve_button)

    def create_result_image(self):
        """Creates the widget to display the result image with white background."""
        self.result_image = WhiteBackgroundImage(
            size_hint=(1, 0.75),
            pos_hint={'x': 0, 'y': 0}
        )
        self.add_widget(self.result_image)

    def update_text_size(self, *args):
        """Centers the text vertically in the TextInput."""
        self.n_input.text_size = (self.n_input.width, None)
        padding_vertical = (self.n_input.height - self.n_input.line_height) / 2
        self.n_input.padding = [0, padding_vertical]

    def increment_value(self, instance):
        """Increments the board size value by 1."""
        try:
            current_value = int(self.n_input.text)
            self.n_input.text = str(current_value + 1)
        except ValueError:
            self.n_input.text = '1'  # Default to 1 if invalid input

    def decrement_value(self, instance):
        """Decrements the board size value by 1, minimum value is 1."""
        try:
            current_value = int(self.n_input.text)
            if current_value > 1:
                self.n_input.text = str(current_value - 1)
        except ValueError:
            self.n_input.text = '1'  # Default to 1 if invalid input

    def select_image(self, instance):
        """Opens a dialog window to select an image."""
        file_path = filechooser.open_file(
            title='Select an image',
            filters=[('Images', '*.png;*.jpg;*.jpeg')]
        )
        if file_path:
            self.selected_image = file_path[0]
            self.image_button.text = 'Image selected'

    def solve_puzzle(self, instance):
        """Sends a request to the API to solve the puzzle and displays the result."""
        if not self.selected_image:
            self.image_button.text = 'Please select an image!'
            return
        try:
            n = int(self.n_input.text)
        except ValueError:
            self.n_input.text = '7'  # Default to 7 if invalid input
            n = 7

        with open(self.selected_image, 'rb') as f:
            img_bytes = f.read()
        encoded_img = base64.b64encode(img_bytes).decode('utf-8')

        data = {
            'n': n,
            'image': encoded_img
        }

        # Send a request to the API
        try:
            response = requests.post('http://localhost:5000/solve', json=data)
            if response.status_code == 200:
                result = response.json()
                solution_img_data = base64.b64decode(result['solution_image'])
                img = CoreImage(io.BytesIO(solution_img_data), ext='png')
                self.result_image.texture = img.texture
            else:
                self.solve_button.text = 'Error solving the puzzle'
        except Exception as e:
            self.solve_button.text = 'Error connecting to the API'
            print(e)

class NQueensApp(App):
    def build(self):
        return MainWidget()

if __name__ == '__main__':
    NQueensApp().run()
