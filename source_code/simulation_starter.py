import logging
import logging.config
from pathlib import Path

from opensc2_gui import OPENSC2_GUI

# from opensc2_gui_simpl import OPENSC2_GUI

config_path = Path(__file__).with_name("logging_opensc2.conf")
logging.config.fileConfig(fname=config_path, disable_existing_loggers=True)

# Get the logger specified in the file, this will be the parent logger.
logger = logging.getLogger("opensc2Logger")

# make an instance of class OPENSC2_GUI (cdp, 12/2020)
gui = OPENSC2_GUI()
# Infinite loop of the main_window to start the program (cdp, 11/2020)
gui.main_window.mainloop()
