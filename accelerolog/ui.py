"""User interface declarations for Accelerolog."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

from kivy.clock import Clock
from kivy.properties import DictProperty, ListProperty, NumericProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.spinner import Spinner
from kivy.uix.togglebutton import ToggleButton

from kivy_garden.graph import Graph, LinePlot

from sensors import SensorSample


MAX_GRAPH_POINTS = 600


@dataclass
class SensorOption:
    """Representation of selectable sensors on the home screen."""

    key: str
    label: str
    available: bool


class SensorTile(BoxLayout):
    """Card-like widget showing live data and a graph for a sensor."""

    sensor_name = StringProperty("")
    display_name = StringProperty("")
    latest_values = ListProperty([0.0, 0.0, 0.0])
    plot_colors = ListProperty(
        [
            (1.0, 0.3, 0.3, 1.0),
            (0.3, 1.0, 0.3, 1.0),
            (0.3, 0.3, 1.0, 1.0),
        ]
    )

    def __init__(self, sensor_name: str, display_name: str, **kwargs) -> None:
        super().__init__(orientation="vertical", padding=8, spacing=6, **kwargs)
        self.sensor_name = sensor_name
        self.display_name = display_name
        self._front_buffer: Deque[Tuple[int, float, float, float]] = deque(maxlen=MAX_GRAPH_POINTS)
        self._back_buffer: List[Tuple[float, float, float]] = []
        self._sample_index = 0
        self._graph = Graph(
            xlabel="muestras",
            ylabel="valor",
            x_ticks_minor=5,
            x_ticks_major=25,
            y_ticks_major=1,
            y_grid_label=True,
            x_grid_label=True,
            padding=10,
            xmin=0,
            xmax=MAX_GRAPH_POINTS,
            ymin=-20,
            ymax=20,
            xmax_trigger=MAX_GRAPH_POINTS,
            background_color=(0.05, 0.05, 0.05, 0.65),
        )
        header = Label(text=display_name, bold=True, size_hint_y=None, height=24)
        self.add_widget(header)
        self.add_widget(self._graph)
        self._plots = [LinePlot(color=color, line_width=1.5) for color in self.plot_colors]
        for plot in self._plots:
            plot.points = []
            self._graph.add_plot(plot)
        self._numeric_label = Label(text="0.00 | 0.00 | 0.00", size_hint_y=None, height=24)
        self.add_widget(self._numeric_label)
        Clock.schedule_interval(self._on_render_tick, 1.0 / 30.0)

    def add_sample(self, sample: SensorSample) -> None:
        """Buffer a sample for later rendering."""
        self._back_buffer.append((sample.ax, sample.ay, sample.az))
        self.latest_values = [sample.ax, sample.ay, sample.az]
        self._numeric_label.text = f"{sample.ax:6.2f} | {sample.ay:6.2f} | {sample.az:6.2f}"

    def _on_render_tick(self, *_args) -> None:
        if not self._back_buffer:
            return
        for values in self._back_buffer:
            self._sample_index += 1
            self._front_buffer.append((self._sample_index, values[0], values[1], values[2]))
        self._back_buffer.clear()
        points = list(self._front_buffer)
        if not points:
            return
        xs = [value[0] for value in points]
        for axis in range(3):
            self._plots[axis].points = [(xs[i], points[i][axis + 1]) for i in range(len(points))]
        self._graph.xmin = max(0, self._sample_index - MAX_GRAPH_POINTS)
        self._graph.xmax = self._sample_index


class HomeScreen(Screen):
    """Initial screen where the user picks sensors and sampling rate."""

    controller = ObjectProperty(None)

    def __init__(self, controller, sensors: Iterable[SensorOption], rates: Iterable[int], **kwargs) -> None:
        super().__init__(name="home", **kwargs)
        self.controller = controller
        self._sensor_options = list(sensors)
        self._selected: Dict[str, ToggleButton] = {}
        layout = BoxLayout(orientation="vertical", padding=16, spacing=12)
        layout.add_widget(Label(text="Selecciona los sensores", size_hint_y=None, height=30))
        sensor_list = BoxLayout(orientation="vertical", spacing=6, size_hint_y=0.6)
        for option in self._sensor_options:
            toggle = ToggleButton(
                text=f"{option.label} ({'Disponible' if option.available else 'No disponible'})",
                state="down" if option.available else "normal",
                disabled=not option.available,
                group=None,
                allow_no_selection=True,
                size_hint_y=None,
                height=40,
            )
            sensor_list.add_widget(toggle)
            self._selected[option.key] = toggle
        layout.add_widget(sensor_list)
        layout.add_widget(Label(text="Tasa de muestreo objetivo", size_hint_y=None, height=30))
        self.rate_spinner = Spinner(text=f"{next(iter(rates))} Hz", values=[f"{rate} Hz" for rate in rates], size_hint_y=None, height=40)
        layout.add_widget(self.rate_spinner)
        button_bar = BoxLayout(size_hint_y=None, height=50, spacing=12)
        start_button = Button(text="Iniciar adquisición", on_release=self._on_start)
        demo_button = Button(text="Modo demo", on_release=lambda *_: self._on_start(demo=True))
        button_bar.add_widget(start_button)
        button_bar.add_widget(demo_button)
        layout.add_widget(button_bar)
        self.add_widget(layout)

    def _on_start(self, *_args, demo: bool = False) -> None:
        selected = [key for key, toggle in self._selected.items() if toggle.state == "down"]
        if not selected:
            self._show_message("Selecciona al menos un sensor disponible.")
            return
        rate_value = int(self.rate_spinner.text.split()[0])
        self.controller.on_start_requested(selected, rate_value, demo)

    def _show_message(self, message: str) -> None:
        content = BoxLayout(orientation="vertical", padding=12, spacing=12)
        content.add_widget(Label(text=message))
        close_btn = Button(text="Cerrar", size_hint_y=None, height=40)
        popup = Popup(title="Atención", content=content, size_hint=(0.6, 0.3))

        def close_popup(*_args) -> None:
            popup.dismiss()

        close_btn.bind(on_release=close_popup)
        content.add_widget(close_btn)
        popup.open()


class AcquisitionScreen(Screen):
    """Screen that renders live data and exposes acquisition controls."""

    controller = ObjectProperty(None)
    sensor_tiles = DictProperty({})

    def __init__(self, controller, **kwargs) -> None:
        super().__init__(name="acquisition", **kwargs)
        self.controller = controller
        self.sensor_tiles = {}
        self._layout = BoxLayout(orientation="vertical", padding=12, spacing=12)
        toolbar = BoxLayout(size_hint_y=None, height=50, spacing=10)
        stop_btn = Button(text="Detener", on_release=lambda *_: self.controller.on_stop_requested())
        save_btn = Button(text="Guardar CSV", on_release=lambda *_: self.controller.on_save_requested())
        share_btn = Button(text="Compartir", on_release=lambda *_: self.controller.on_share_requested())
        calib_btn = Button(text="Calibrar", on_release=lambda *_: self.controller.on_open_calibration())
        toolbar.add_widget(stop_btn)
        toolbar.add_widget(save_btn)
        toolbar.add_widget(share_btn)
        toolbar.add_widget(calib_btn)
        self._layout.add_widget(toolbar)
        self._tiles_wrapper = BoxLayout(orientation="vertical", spacing=12)
        self._layout.add_widget(self._tiles_wrapper)
        self.add_widget(self._layout)

    def configure_sensors(self, sensors: Iterable[SensorOption]) -> None:
        self._tiles_wrapper.clear_widgets()
        self.sensor_tiles.clear()
        for option in sensors:
            tile = SensorTile(sensor_name=option.key, display_name=option.label)
            self.sensor_tiles[option.key] = tile
            self._tiles_wrapper.add_widget(tile)

    def process_sample(self, sample: SensorSample) -> None:
        tile = self.sensor_tiles.get(sample.sensor_name)
        if not tile:
            return
        tile.add_sample(sample)


class CalibrationScreen(Screen):
    """Guided calibration routine."""

    controller = ObjectProperty(None)
    status_text = StringProperty("Coloca el dispositivo sobre una superficie estable y presiona 'Iniciar'.")
    progress = NumericProperty(0.0)

    def __init__(self, controller, **kwargs) -> None:
        super().__init__(name="calibration", **kwargs)
        self.controller = controller
        layout = BoxLayout(orientation="vertical", padding=20, spacing=12)
        self._status_label = Label(text=self.status_text, halign="center", valign="middle")
        layout.add_widget(self._status_label)
        self._progress_bar = ProgressBar(max=100, value=self.progress)
        layout.add_widget(self._progress_bar)
        button_bar = BoxLayout(size_hint_y=None, height=50, spacing=12)
        start_btn = Button(text="Iniciar", on_release=lambda *_: self.controller.on_calibration_start())
        cancel_btn = Button(text="Cancelar", on_release=lambda *_: self.controller.on_calibration_cancel())
        button_bar.add_widget(start_btn)
        button_bar.add_widget(cancel_btn)
        layout.add_widget(button_bar)
        self.add_widget(layout)

    def update_status(self, text: str, progress: float) -> None:
        self.status_text = text
        self.progress = progress
        self._status_label.text = text
        self._progress_bar.value = progress


class AccelerologScreenManager(ScreenManager):
    """Root widget that wires all screens together."""

    def __init__(self, controller, sensor_options: Iterable[SensorOption], rates: Iterable[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.controller = controller
        self.home_screen = HomeScreen(controller=controller, sensors=sensor_options, rates=rates)
        self.acquisition_screen = AcquisitionScreen(controller=controller)
        self.calibration_screen = CalibrationScreen(controller=controller)
        self.add_widget(self.home_screen)
        self.add_widget(self.acquisition_screen)
        self.add_widget(self.calibration_screen)

    def show_home(self) -> None:
        self.current = "home"

    def show_acquisition(self) -> None:
        self.current = "acquisition"

    def show_calibration(self) -> None:
        self.current = "calibration"


class UIController:
    """Bridge between Kivy UI widgets and the application controller."""

    def __init__(
        self,
        sensor_options: Iterable[SensorOption],
        rates: Iterable[int],
        callbacks: Dict[str, Callable[..., None]],
    ) -> None:
        self._callbacks = callbacks
        self._sensor_options = list(sensor_options)
        self._rates = list(rates)
        self.root = AccelerologScreenManager(controller=self, sensor_options=self._sensor_options, rates=self._rates)

    # Interactions requested by widgets -------------------------------------------------

    def on_start_requested(self, sensors: Iterable[str], target_rate: int, demo: bool = False) -> None:
        self._callbacks["start"](list(sensors), target_rate, demo)
        selected_options = [option for option in self._sensor_options if option.key in sensors]
        self.root.acquisition_screen.configure_sensors(selected_options)
        self.root.show_acquisition()

    def on_stop_requested(self) -> None:
        self._callbacks["stop"]()
        self.root.show_home()

    def on_save_requested(self) -> None:
        self._callbacks["save"]()

    def on_share_requested(self) -> None:
        self._callbacks["share"]()

    def on_open_calibration(self) -> None:
        self.root.show_calibration()

    def on_calibration_start(self) -> None:
        self._callbacks["calibrate_start"](self._update_calibration_status)

    def on_calibration_cancel(self) -> None:
        self._callbacks["calibrate_cancel"]()
        self.root.show_acquisition()

    # Hooks used by the application core ------------------------------------------------

    def update_acquisition_sample(self, sample: SensorSample) -> None:
        self.root.acquisition_screen.process_sample(sample)

    def notify_share_result(self, success: bool) -> None:
        if not success:
            popup = Popup(title="Compartir", content=Label(text="No se pudo compartir el archivo CSV."), size_hint=(0.6, 0.3))
            popup.open()

    def _update_calibration_status(self, message: str, progress: float) -> None:
        self.root.calibration_screen.update_status(message, progress)

    def calibration_completed(self, message: str) -> None:
        popup = Popup(title="Calibración", content=Label(text=message), size_hint=(0.7, 0.3))
        popup.open()
        self.root.show_acquisition()


def build_ui(
    sensor_options: Iterable[SensorOption],
    rates: Iterable[int],
    callbacks: Dict[str, Callable[..., None]],
) -> UIController:
    """Factory helper to create the root UI controller."""
    return UIController(sensor_options=sensor_options, rates=rates, callbacks=callbacks)

