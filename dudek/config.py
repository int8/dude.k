import os
from dotenv import load_dotenv

load_dotenv()

VIDEOS_DATASET_PATH = os.getenv("VIDEOS_DATASET_PATH")
SN_BAS_2025_DATASET_PATH = os.getenv("SN_BAS_2025_DATASET_PATH")


EXPERIMENTS_RANDOM_SEED = os.getenv("EXPERIMENTS_RANDOM_SEED", 42)
TEST_SET_CHALLENGE_SEED = 116
_FUCKED_UP_VIDEOS = [
    "italy_serie-a/2016-2017/2016-10-02 - 21-45 AS Roma 2 - 1 Inter/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-10-02 - 21-45 AS Roma 2 - 1 Inter/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-11 - 16-00 AC Milan 0 - 1 Udinese/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-11 - 16-00 AC Milan 0 - 1 Udinese/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-08-28 - 21-45 Cagliari 2 - 2 AS Roma/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-08-28 - 21-45 Cagliari 2 - 2 AS Roma/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-08-27 - 21-45 Napoli 4 - 2 AC Milan/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-08-27 - 21-45 Napoli 4 - 2 AC Milan/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-25 - 13-30 Torino 3 - 1 AS Roma/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-25 - 13-30 Torino 3 - 1 AS Roma/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-20 - 21-45 AC Milan 2 - 0 Lazio/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-20 - 21-45 AC Milan 2 - 0 Lazio/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-16 - 21-45 Sampdoria 0 - 1 AC Milan/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-16 - 21-45 Sampdoria 0 - 1 AC Milan/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-21 - 21-45 AS Roma 4 - 0 Crotone/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-21 - 21-45 AS Roma 4 - 0 Crotone/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-18 - 21-45 Fiorentina 1 - 0 AS Roma/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-18 - 21-45 Fiorentina 1 - 0 AS Roma/2_720p.mkv",
]

DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE", "cuda")
