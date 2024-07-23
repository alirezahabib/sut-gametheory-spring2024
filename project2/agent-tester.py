import random as rand
from agent import Agent99109393
from models import Log, Round

def generate_random_log(player_id: int) -> Log:
    player_log: Log = Log(player_id, [])
    opponents = rand.sample(range(3, 50), rounds)
    for i, id in enumerate(opponents):
        if rand.randint(0, 1) == 0:
            proposer = 1
            responder = id
        else:
            proposer = id
            responder = 1
        offer = rand.randint(0, 100)
        accepted = rand.choice([True, False])
        r = Round(i + 1, proposer, responder, offer, accepted)
        player_log.games.append(r)
    return player_log

if __name__ == "__main__":
    rounds = 10
    player_log = generate_random_log(1)
    opponent_log = generate_random_log(2)

    is_proposer = rand.choice([True, False])
    agent = Agent99109393(player_log, opponent_log, is_proposer)
    if is_proposer:
        print(agent.proposer_strategy())
    else:
        print(agent.responder_strategy(rand.randint(0, 100)))
