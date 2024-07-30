import numpy as np
import datetime
from datetime import datetime, timezone
from data_loader import EventsDataset


class ExampleDataset(EventsDataset):

    def __init__(self, split, data_dir=None):
        super(ExampleDataset, self).__init__()

        if split == 'train':
            time_start = 0
            time_end = datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=12).toordinal()
        elif split == 'test':
            time_start = datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=15).toordinal()
            time_end = datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=0).toordinal()
        else:
            raise ValueError('invalid split', split)

        self.FIRST_DATE = datetime(2016, 1, 4, tzinfo=self.TZ)

        self.TEST_TIMESLOTS = [
            datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=6),
            datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=9),
            datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=12),
            datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=15),
            datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=18),
            datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=21),
            datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=6),
            datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=9),
            datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=12),
            datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=15),
            datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=18),
            datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=21),
            datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=6),
            datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=9),
            datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=12),
            datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=15),
            datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=18),
            datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=21),
            datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=0)
        ]




        self.A_initial = np.loadtxt('data_common_slots_csv/adj_matrices/Adj_0.csv', delimiter=',')
        self.A_last = np.loadtxt('data_common_slots_csv/adj_matrices/Adj_17.csv', delimiter=',')
        self.N_nodes = self.A_last.shape[1]

        print('\nA_initial', np.sum(self.A_initial))
        print('A_last', np.sum(self.A_last), '\n')

        
        all_events = []
        edges_info = np.loadtxt('non_directed_graph_info/ID_edges_info.csv', delimiter=',')
        all_events = edges_info.tolist()

        self.event_types = ['communication event']

        self.all_events = sorted(all_events, key=lambda t: t[3].timestamp())
        print('\n%s' % split.upper())
        print('%d events between %d users loaded' % (len(self.all_events), self.N_nodes))
        # print('%d communication events' % (len([t for t in self.all_events if t[2] == 1])))
        # print('%d assocition events' % (len([t for t in self.all_events if t[2] == 0])))

        self.event_types_num = {'association event': 0}
        k = 1  # k >= 1 for communication events
        for t in self.event_types:
            self.event_types_num[t] = k
            # k += 1

        self.n_events = len(self.all_events)

    def get_Adjacency(self, multirelations=False):
        if multirelations:
            print('warning: this dataset has only one relation type, so multirelations are ignored')
        return self.A_initial, ['association event'], self.A_last
