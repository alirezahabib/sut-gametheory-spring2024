from dataclasses import dataclass

@dataclass
class Round:
    week: int
    proposer_id: int
    responder_id: int
    offer: int
    accepted: bool

@dataclass
class Log:
    player_id: int
    games: list[Round]
    