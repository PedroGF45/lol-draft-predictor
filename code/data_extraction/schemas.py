import pyarrow as pa

PLAYERS_SCHEMA = pa.schema([
    ('puuid', pa.string())
])

MATCHES_SCHEMA = pa.schema([
    ('match_id', pa.string())
])

MATCH_SCHEMA = pa.schema([
    ('match_id', pa.string()),
    ('queue_id', pa.int32()),
    ('game_version', pa.string()),
    ('game_duration', pa.int32()),
    ('team1_win', pa.bool_()),

    # bans and picks
    ('team1_ban1', pa.int32()),
    ('team1_ban2', pa.int32()),
    ('team1_ban3', pa.int32()),
    ('team1_ban4', pa.int32()),
    ('team1_ban5', pa.int32()),
    ('team2_ban1', pa.int32()),
    ('team2_ban2', pa.int32()),
    ('team2_ban3', pa.int32()),
    ('team2_ban4', pa.int32()),
    ('team2_ban5', pa.int32()),
    ('team1_pick_top', pa.int32()),
    ('team1_pick_jungle', pa.int32()),
    ('team1_pick_mid', pa.int32()),
    ('team1_pick_adc', pa.int32()),
    ('team1_pick_support', pa.int32()),
    ('team2_pick_top', pa.int32()),
    ('team2_pick_jungle', pa.int32()),
    ('team2_pick_mid', pa.int32()),
    ('team2_pick_adc', pa.int32()),
    ('team2_pick_support', pa.int32())
])

PLAYER_HISTORY_SCHEMA = pa.schema([
    ('match_id', pa.string()),
    ('puuid', pa.string()),
    ('team_id', pa.int32()),
    ('role', pa.string()),
    ('champion_id', pa.int32()),

    # state of the summoner
    ('summoner_level', pa.int32()),
    ('champion_mastery_level', pa.int32()),
    ('champion_total_mastery_score', pa.int32()),
    ('ranked_tier', pa.string()),
    ('ranked_rank', pa.string()),
    ('ranked_league_points', pa.int32()),

    # kpis
    ('win_rate', pa.float32()),
    ('average_kills', pa.float32()),
    ('average_deaths', pa.float32()),
    ('average_assists', pa.float32()),
    ('average_cs', pa.float32()),
    ('average_gold_earned', pa.float32()),
    ('average_damage_dealt', pa.float32()),
    ('average_damage_taken', pa.float32()),
    ('average_vision_score', pa.float32()),
    ('average_healing_done', pa.float32()),
    ('kda_ratio', pa.float32()),
    ('cs_per_minute', pa.float32()),
    ('gold_per_minute', pa.float32()),
    ('damage_per_minute', pa.float32()),
    ('vision_score_per_minute', pa.float32()),
    ('healing_per_minute', pa.float32())
])