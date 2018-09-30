import sched
import time
import threading
import pandas as pd

def load_csv_np(filename, storage, y_column=None):
    """Convert CSV file to a tensorflow tensor and store it in X and Y.

    Parameters
    -----------
        filename : str
            The file to load

        storage : AutoLoader
            Class to store data

        y_column : str or int, default None
            Column from which target is retrieved. If None, no target is set

    Returns
    --------
        A tensorflow tensor

    """
    global X, Y
    df = pd.read_csv(filename)

    if y_column is not None:
        y = df[y_column].values
    else:
        y = None

    x = df.values

    storage.update(x, y)


# Better put this in a util file. Useless for so few code though
def get_repeater(nb_second, action, action_args=[], action_kwargs={}, delay_first=False):
    """Get a repeater.

    Parameters
    -----------
        nb_second : float
            Number of seconds between two calls

        action : function
            Function to call every nb_second seconds

        action_args : list, default None
            Arguments for action

        action_kwargs : dic, default None
            Argument dic for action

        delay_first : bool, default False
            Whether you first call action or you first delay

    Returns
    --------
        A sched.scheduler. You just have to call run on it so it starts. If you want to keep
        control, don't forget to start it in a thread.
    """
    s = sched.scheduler(time.time, time.sleep)

    def repeat(nb_second, action, action_args, action_kwargs):
        """Repeat function."""
        action(*action_args, **action_kwargs)
        s.enter(nb_second, 1, repeat, (nb_second, action, action_args, action_kwargs))

    if delay_first:
        s.enter(nb_second, 1, repeat, (nb_second, action, action_args, action_kwargs))
    else:
        s.enter(0, 1, repeat, (nb_second, action, action_args, action_kwargs))
    return s


class AutoLoader(object):
    """An auto loader of csv file.

    Attributes
    -----------
        x : np array
        y : np array
        updated : bool
            Has it been updated (if so, last row is new)
    """

    def __init__(self, filename, repeat_every, delay_first=False):
        """Init autoloader.

        Parameters
        -----------
            filename : str
                File to load

            repeat_every : float
                Number of seconds between two loads

            delay_first : bool, default False
                Whether it starts by doing a delay or by loading
        """
        self.scheduler = get_repeater(repeat_every, load_csv_np, delay_first=delay_first, action_args=[filename, self])
        self.thread = threading.Thread(target=self.scheduler.run)
        self.x = None
        self.y = None
        self.updated = False

    def run(self):
        self.thread.start()

    def update(self, x, y):
        if x[-1, 1] != self.x[-1, 1]:
            self.x = x
            self.y = y
            self.updated = True

    def cancel(self):
        self.scheduler.cancel()

    def get(self):
        self.updated = False
        return self.x, self.y


filename = "eurusd.csv"
nb_minutes = 5
data = AutoLoader(filename, nb_minutes*60)
data.run()

waiter = threading.Condition()
while True:
    waiter.wait_for(data.updated)
    # process x
