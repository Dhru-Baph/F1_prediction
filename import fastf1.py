import fastf1
fastf1.Cache.enable_cache("cache")  # Enables local caching for better performance

session = fastf1.get_session(2023, "Bahrain", "Q")
session.load()
print(session.laps.head())  # Prints the first few laps of the session
