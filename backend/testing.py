from catalogs import load_events_catalog

events_df = load_events_catalog()
print(events_df.shape)
print(type(events_df['Categories'].iloc[0]))  # must print <class 'list'>, not <class 'str'>