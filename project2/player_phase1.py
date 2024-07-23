class UGPlayerPhase1_99109393:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """ Reset any state variables if necessary. Called before starting a new game. """
        self.acceptance_threshold = 50  # Start with a fair split assumption
        self.history = []

    def proposer_strategy(self, round_number: int) -> int:
        """
        Define the strategy for the proposer.

        Args:
            round_number (int): The current round number (1 to 100).

        Returns:
            int: The amount offered to the responder (0 to 100).
        """
        if round_number < 10:
            return 50  # Start with a fair offer

        # Adjust based on the opponent's past behavior
        last_offers = [round[1] for round in self.history[-10:]]

        average_acceptance = sum(last_offers) / len(last_offers)
        offer = max(30, min(70, int(average_acceptance * 0.8)))  # Adjust based on average acceptance, within a range

        return offer

    def responder_strategy(self, round_number: int, offer: int) -> bool:
        """
        Define the strategy for the responder.

        Args:
            round_number (int): The current round number (1 to 100).
            offer (int): The amount offered by the proposer (0 to 100).

        Returns:
            bool: True if the offer is accepted, False otherwise.
        """
        return offer >= self.acceptance_threshold

    def result(self, round_number: int, score: int) -> None:
        """
        Receive the result of the round.

        Args:
            round_number (int): The round number (1 to 100).
            score (int): The score for the round.
        """
        self.history.append((round_number, score))
        # Adjust acceptance threshold based on the history
        if len(self.history) >= 10:
            recent_results = self.history[-10:]
            accepted_offers = [offer for round_num, offer in recent_results if offer >= self.acceptance_threshold]
            if len(accepted_offers) / 10 < 0.6:
                self.acceptance_threshold -= 5  # Be more lenient if less than 60% offers are accepted
            else:
                self.acceptance_threshold += 5  # Be stricter if more than 60% offers are accepted

        # Ensure the acceptance threshold stays within 30 to 70 range
        self.acceptance_threshold = max(30, min(70, self.acceptance_threshold))

        print(f"Round {round_number}: {score}")
