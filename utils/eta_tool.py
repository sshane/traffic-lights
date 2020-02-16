import datetime


class ETATool:
    def __init__(self):  # only supports up to minutes
        self.start_time = 0
        self.max_progress = 0
        self.progress = 0
        self.time = 0
        self.etr = 0  # in seconds, estimated time remained
        self.seconds = 60

    def init(self, t, max_progress):
        self.start_time = t
        self.max_progress = max_progress

    def log(self, progress, t):
        self.progress = progress
        self.time = t

    @property
    def get_eta(self):
        elapsed =self.time - self.start_time
        etr = self.max_progress * (elapsed / (self.progress + 1)) - elapsed

        sec_string = ''
        min_string = ''
        etr = datetime.timedelta(seconds=round(etr))
        etr = str(etr).split(':')[-2:]
        minutes = int(etr[0])
        seconds = int(etr[1])

        sec_string += '{} second'.format(seconds)
        min_string += '{} minute'.format(minutes)

        if seconds != 1:
            sec_string += 's'
        if minutes != 1:
            min_string += 's'

        etr_string = []
        if minutes != 0:
            etr_string.append(min_string)
        etr_string.append(sec_string)
        etr_string = ', '.join(etr_string)

        return etr_string
