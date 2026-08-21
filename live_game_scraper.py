import requests
from datetime import date

BASE_URL = "https://statsapi.mlb.com/api/v1"


def get_current_lineups(team=None):
    today = date.today().isoformat()

    # Find today's games
    schedule_url = f"{BASE_URL}/schedule"

    params = {
        "sportId": 1,
        "date": today,
        "hydrate": "team"
    }

    response = requests.get(schedule_url, params=params)
    response.raise_for_status()
    schedule = response.json()

    game = None

    for day in schedule.get("dates", []):
        for g in day["games"]:

            away = g["teams"]["away"]["team"]
            home = g["teams"]["home"]["team"]

            away_name = away["name"]
            home_name = home["name"]

            away_abbr = away.get("abbreviation", "")
            home_abbr = home.get("abbreviation", "")

            if team is not None:
                search = team.lower()

                matches = (
                    search in away_name.lower()
                    or search in home_name.lower()
                    or search == away_abbr.lower()
                    or search == home_abbr.lower()
                )

                if not matches:
                    continue

            # If no team was specified, only select live games
            status = g["status"]["abstractGameState"]

            if team is None and status != "Live":
                continue

            game = g
            break

        if game:
            break

    if game is None:
        raise ValueError(f"No game found for {team} today.")

    game_pk = game["gamePk"]

    # IMPORTANT: v1.1 for the live feed
    feed_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

    response = requests.get(feed_url)
    response.raise_for_status()

    data = response.json()

    result = {}

    for side in ["away", "home"]:

        team_data = data["liveData"]["boxscore"]["teams"][side]

        team_name = team_data["team"]["name"]

        # Current batting order
        lineup = []

        for player_id in team_data.get("battingOrder", []):

            player = team_data["players"][f"ID{player_id}"]
            person = player["person"]

            lineup.append(
                [person['id'], person['fullName']]
            )

        # Current pitcher
        pitchers = team_data.get("pitchers", [])

        current_pitcher = None

        if pitchers:
            pitcher_id = pitchers[-1]

            pitcher_data = team_data["players"][f"ID{pitcher_id}"]
            person = pitcher_data["person"]

            current_pitcher = (
                person['id'], person['fullName']
            )

        result[side] = {
            "team": team_name,
            "lineup": lineup,
            "pitcher": current_pitcher
        }

    return result