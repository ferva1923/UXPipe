import pandas as pd
import matplotlib.pyplot as plt


class UXData:
    def __init__(self, filepath, type):
        self.filepath = filepath
        self.type = type
        """
        Read data from a file.

        Args:
            filepath (str): The path to the file.
            type (str): The type of the file ('csv' or 'json').

        Returns:
            pandas.DataFrame: The loaded data.
        """
        if type == 'csv':
            data = pd.read_csv(filepath)
        elif type == 'json':
            data = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file type")
        data = pd.DataFrame(data)
        self.data = data
        return None

#Define the functions that will be used in the load function
        


    def standardize_time(self):
        """
        Standardize the timestamp column to datetime format.
        """
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        return self

    def rename_columns(self, original_columns=['user_id','session_id','timestamp','event_type']):
        self.data = self.data.rename(columns= {original_columns[0]:'user_id', original_columns[1]:'session_id', original_columns[2]:'timestamp', original_columns[3]:'event_type'})
        return self

    
        

    # Compute new metrics
    def events_per_session(self):
        """
        Calculate the number of events per session.

        Returns:
            data with new column events_per_session
        """
        events_count = self.data.groupby('session_id').count()
        events_count['events_per_session'] = events_count['event_type']
        self.data = self.data.merge(events_count['events_per_session'], on='session_id')
        return self
        
    def session_duration(self):
        """
        Calculate session duration for each session.

        Returns:
            data with new column session duration
        """
        session_times = self.data.groupby('session_id')['timestamp'].agg(['min', 'max'])
        session_times['session_duration'] = (session_times['max'] - session_times['min']).dt.total_seconds()
        self.data = self.data.merge(session_times['session_duration'], on='session_id', how='left')
        return self

    def add_metrics(self):
        """Adds the new metrics to the dataset"""
        self = self.events_per_session()
        self = self.session_duration()
        return self



    #Visualization functions
    def plot_events_per_session(self, figsize=(10, 6)):
        """
        Plot the distribution of events per session.
        """
        plt.figure(figsize=figsize)
        self.data['events_per_session'].hist(bins=30)
        plt.title('Distribution of Events per Session')
        plt.xlabel('Number of Events')
        plt.ylabel('Frequency')
        plt.show()


    def plot_event_types(self, figsize=(10, 6)):
        """
        Plot the distribution of event types.
        """
        plt.figure(figsize=figsize)
        self.data['event_type'].value_counts().plot(kind='bar')
        plt.title('Distribution of Event Types')
        plt.xlabel('Event Type')
        plt.ylabel('Frequency')
        plt.show()
