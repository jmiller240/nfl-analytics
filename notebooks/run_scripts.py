

from resources.game_review import main as game_review_main
from resources.game_preview import main as game_preview_main



''' Season / Week '''
SEASON = 2026
WEEK = 2


''' Scripts '''
GAME_REVIEW = True
GAME_PREVIEW = True


def run():
    if GAME_REVIEW:
        game_review_main(season=SEASON, week=WEEK)

    if GAME_PREVIEW:
        game_preview_main(season=SEASON, week=WEEK)


run()