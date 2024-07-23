from models import Round, Log

class Agent99109393:
    def __init__(self, player_log: Log, opponent_log: Log, proposer: bool) -> None:
        self.player_log = player_log
        self.opponent_log = opponent_log
        self.proposer = proposer
        self.acceptance_threshold = 50  # Start with a fair split assumption
        self.history = []

    def proposer_strategy(self) -> int:
        if len(self.history) < 10:
            return 50  # Start with a fair offer

        # Analyze the opponent's behavior from their log
        last_offers = [round.offer for round in self.opponent_log.games[-10:] if round.responder == self.opponent_log.player_id]
        if len(last_offers) > 0:
            average_acceptance = sum(last_offers) / len(last_offers)
            offer = max(30, min(70, int(average_acceptance * 0.8)))  # Adjust based on average acceptance, within a range
        else:
            offer = 50  # Fallback to a fair offer if no valid data

        return offer

    def responder_strategy(self, offer: int) -> bool:
        # Consider the opponent's behavior from their log
        opponent_offers = [round.offer for round in self.opponent_log.games if round.proposer_id == self.opponent_log.player_id]
        if opponent_offers:
            average_offer = sum(opponent_offers) / len(opponent_offers)
            self.acceptance_threshold = max(30, min(70, int(average_offer * 0.8)))

        if offer >= self.acceptance_threshold:
            return True
        else:
            return False

    def result(self, result: Round) -> None:
        self.history.append(result)
        # Adjust acceptance threshold based on the history
        if len(self.history) >= 10:
            recent_results = self.history[-10:]
            accepted_offers = [round.offer for round in recent_results if round.responder == self.player_log.player_id and round.accepted]
            if len(accepted_offers) / 10 < 0.6:
                self.acceptance_threshold -= 5  # Be more lenient if less than 60% offers are accepted
            else:
                self.acceptance_threshold += 5  # Be stricter if more than 60% offers are accepted

        # Ensure the acceptance threshold stays within 30 to 70 range
        self.acceptance_threshold = max(30, min(70, self.acceptance_threshold))

    def slogan(self) -> str:
        return "I am a great agent!"
