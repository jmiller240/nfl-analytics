

from resources.game_review import main as game_review_main
from resources.game_preview import main as game_preview_main



''' Season / Week '''
WEEK = 18

''' Scripts '''
GAME_REVIEW = False
GAME_PREVIEW = True


def run():
    if GAME_REVIEW:
        game_review_main(season=2025, week=WEEK)

    if GAME_PREVIEW:
        game_preview_main(season=2025, week=WEEK)


run()