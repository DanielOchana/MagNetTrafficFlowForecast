import numpy as np
import datetime
from datetime import datetime, timezone
from data_loader import EventsDataset
import pandas


class TrafficDataset(EventsDataset):

    def __init__(self, split, data_dir=None):
        super(TrafficDataset, self).__init__()

        if split == 'train':
            time_start = 0
            time_end = datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=12).toordinal()
        elif split == 'test':
            time_start = datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=15).toordinal()
            time_end = datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=23,minute=59, second=59).toordinal()
        else:
            raise ValueError('invalid split', split)

        self.FIRST_DATE = datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=6)

        self.TEST_TIMESLOTS = [
            # datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=6),
            # datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=9),
            # datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=12),
            # datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=15),
            # datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=18),
            # datetime(2016, 1, 4, tzinfo=self.TZ).replace(hour=21),
            # datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=6),
            # datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=9),
            # datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=12),
            # datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=15),
            # datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=18),
            # datetime(2016, 1, 5, tzinfo=self.TZ).replace(hour=21),
            # datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=6),
            # datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=9),
            # datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=12),
            datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=15),
            datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=18),
            datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=21),
            datetime(2016, 1, 6, tzinfo=self.TZ).replace(hour=23,minute=59,second=59)
        ]




        self.A_initial = pandas.read_csv('../data_common_slots_csv/adj_matrices/Adj_0.csv', delimiter=',')
        self.A_last = pandas.read_csv('../data_common_slots_csv/adj_matrices/Adj_17.csv', delimiter=',')
        self.N_nodes = self.A_last.shape[1]
        self.A_initial = np.array(self.A_initial)
        self.A_last = np.array(self.A_last)

        print('\nA_initial', np.sum(self.A_initial))
        print('A_last', np.sum(self.A_last), '\n')
        print(self.A_initial.shape, self.A_last.shape)
        
        all_events = []
        # edges_info = np.loadtxt('../non_directed_graph_info/ID_edges_info.csv', delimiter=',')
        csv = pandas.read_csv('../non_directed_graph_info/edges_info.csv')

        to_date2 = lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        all_events = [(data[0]-1, data[1]-1,'assocition event', to_date2(data[2])) for data in csv.values]

        print('all_events', len(all_events), all_events[0])
        self.event_types = ['assocition event']

        self.all_events = sorted(all_events, key=lambda t: t[3].timestamp())
        self.all_events = sorted(all_events, key=lambda t: t[3].timestamp())
        self.all_events_train = [event for event in self.all_events if event[3] < datetime(2016, 1, 6, 12, 0, 0, tzinfo=self.TZ)]
        self.all_events_test = [event for event in self.all_events if event[3] >= datetime(2016, 1, 6, 12, 0, 0, tzinfo=self.TZ)]
        print('\n%s' % split.upper())

        if  split == 'train':
            self.all_events = self.all_events_train
        else :
            self.all_events = self.all_events_test  
        print('\n%s' % split.upper())
        print('%d events between %d users loaded' % (len(self.all_events), self.N_nodes))
        # print('%d communication events' % (len([t for t in self.all_events if t[2] == 1])))
        # print('%d assocition events' % (len([t for t in self.all_events if t[2] == 0])))

        # self.event_types_num = {'communication event': 0}
        # k = 1  # k >= 1 for communication events
        # for t in self.event_types:
        #     self.event_types_num[t] = k
        #     # k += 1
        self.event_types_num = {event: idx for idx, event in enumerate(self.event_types)}

        self.n_events = len(self.all_events)

    def get_Adjacency(self, multirelations=False):
        if multirelations:
            print('warning: this dataset has only one relation type, so multirelations are ignored')
        return self.A_initial, ['assocition event'], self.A_last
