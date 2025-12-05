import pyarrow as pa

PLAYERS_SCHEMA = pa.schema([
    ('puuid', pa.string())
])

MATCHES_SCHEMA = pa.schema([
    ('match_id', pa.string())
])