import pandas as pd

class UXPipe:
    def __init__(self, filepath, type):
        
        if type == 'csv':
            data = pd.read_csv(filepath)
        elif type == 'json':
            data = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file type")
        data = pd.DataFrame(data)
        self.data = data
        # Standardize time
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        return None
    

    def declare_columns(self, columns=['user_id','session_id','timestamp','event_type']):
        self.usercol = columns[0]
        self.sessioncol = columns[1]
        self.timecol = columns[2]
        self.eventcol = columns[3]
        return self
    
    def session_metrics(self):
        self.sessiondf = pd.DataFrame()
        self.sessiondf[self.sessioncol] = self.data[self.sessioncol].unique()

        session_times = self.data.groupby([self.sessioncol,self.usercol])[self.timecol].agg(['min', 'max'])
        print(session_times.head())
        session_times['duration'] = (session_times['max'] - session_times['min']).dt.total_seconds()
        self.sessiondf = self.sessiondf.merge(session_times['duration'], on=self.sessioncol, how='left')

      
            
        events_count = self.data.groupby([self.sessioncol,self.usercol]).count()
        events_count['events_per_session'] = events_count[self.eventcol]
        self.sessiondf = self.sessiondf.merge(events_count['events_per_session'], on=self.sessioncol, how='left')
        

        #event counts
        self.events = self.data[self.eventcol].unique()
        for event in self.events:

            event_count = self.data[self.data[self.eventcol] == event].groupby(self.sessioncol).count()
            event_count[event+'_count'] = event_count[self.eventcol]
            self.sessiondf = self.sessiondf.merge(event_count[event+'_count'], on=self.sessioncol, how='left')

        self.sessiondf = self.sessiondf.merge( self.data[[self.sessioncol, self.usercol]].drop_duplicates(), on=self.sessioncol, how='left')

        return self
    
    def user_metrics(self):

        self.userdf = pd.DataFrame()
        self.userdf[self.usercol] = self.data[self.usercol].unique()


        #sessions per user
        sessions_per_user = self.data.groupby(self.usercol).nunique()[self.sessioncol]
        sessions_per_user.name = 'sessions_per_user'
        self.userdf = self.userdf.merge(sessions_per_user, on=self.usercol, how='left')

        #total time
        user_times = self.sessiondf.groupby(self.usercol)['duration'].sum()
        user_times.name = 'total_time'
        self.userdf = self.userdf.merge(user_times, left_on=self.usercol, right_index=True, how='left')

        #total events
        total_events = self.sessiondf.groupby(self.usercol)['events_per_session'].sum()
        total_events.name = 'total_events'
        self.userdf = self.userdf.merge(total_events, left_on=self.usercol, right_index=True, how='left')
        self.events_count = [event+'_count' for event in self.events]
        for event in self.events_count:
            event_count = self.sessiondf.groupby(self.usercol)[event].sum()
            event_count.name = event
            self.userdf = self.userdf.merge(event_count, left_on=self.usercol, right_index=True, how='left')

        #average session duration
        avg_session_duration = self.sessiondf.groupby(self.usercol)['duration'].mean()
        avg_session_duration.name = 'avg_session_duration'
        self.userdf = self.userdf.merge(avg_session_duration, left_on=self.usercol, right_index=True, how='left')

        return self
    
    def new_metrics(self,col):
        data = pd.DataFrame()
        
        setattr(self, col+'df', data)
        
        return self

    def loadnewdata(self, filepath, type, name):
        if type == 'csv':
            data = pd.read_csv(filepath)
        elif type == 'json':
            data = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file type")
        data = pd.DataFrame(data)
        setattr(self, name, data)

        return self
    
    def getGraph(self):
        class UXGraph:
            def __init__(self):
                self.nodes = UXPipe.events
                self.edges = {}
                for session, user in UXPipe.sessiondf[[UXPipe.sessioncol, UXPipe.usercol]].drop_duplicates().values:
                    session_events = UXPipe.data[UXPipe.data[UXPipe.sessioncol] == session][UXPipe.eventcol].values
                    for i in range(len(session_events)-1):
                        edge = (session_events[i], session_events[i+1])
                        if edge in self.edges:
                            self.edges[edge] += 1
                        else:
                            self.edges[edge] = 1
                return self
        return self
    
    class UXGraph:
        def seeGraph(self):
            print("Nodes:", self.nodes)
            print("Edges:", self.edges)

    

filepath = 'C:\\Users\\ferva\\OneDrive\\UX_Project\\modules\\large_sample_ux_dataset.csv'
uxLib = UXPipe(filepath, 'csv')
uxLib.declare_columns().session_metrics().user_metrics().getGraph()

print(uxLib.sessiondf.head())
print(uxLib.userdf.head())
print(uxLib.nodes)
print(uxLib.edges)
