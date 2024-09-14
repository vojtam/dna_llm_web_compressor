from typing import Dict, List, Tuple, Union


class SymbolFrequencyTable:
    def __init__(
        self,
        symbols: List[Union[int, str]],
        freq: Union[None, List[int]] = None,
        probs: Union[None, List[float]] = None,
        scale_factor: int = 4096,
    ):
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.scale_factor = scale_factor
        self.freq: Dict[Union[int, str], int] = {}
        self.prob: Dict[Union[int, str], float] = {}
        self.cumulative_frequency: Dict[Union[int, str], Tuple[int, int]] = {}
        self.freq_total: int = 0

        if probs is not None:
            self._init_from_probs(probs)
        elif freq is not None:
            self._init_from_freq(freq)
        else:
            self._init_uniform()

    def _init_uniform(self) -> None:
        self.freq = {symbol: 1 for symbol in self.symbols}
        self.prob = {symbol: 1 / self.num_symbols for symbol in self.symbols}
        self.cumulative_frequency = self.update_cum_freq()

    def _init_from_freq(self, freq: List[int]) -> None:
        """Initialize the table from frequencies."""
        self.freq = dict(zip(self.symbols, freq))
        self.prob = {
            symbol: freq / sum(self.freq.values()) for symbol, freq in self.freq.items()
        }
        self.update_cum_freq()

    def _init_from_probs(self, probs: List[float]) -> None:
        """Initialize the table from probabilities."""
        self.prob = dict(zip(self.symbols, probs))
        self.freq = {
            symbol: round(self.scale_factor * prob)
            for symbol, prob in self.prob.items()
        }
        cumulative_frequency = {}
        cumsum = 0
        for symbol, freq in self.freq.items():
            cumulative_frequency[symbol] = (cumsum, cumsum + freq)
            cumsum += freq
        self.cumulative_frequency = cumulative_frequency

    def get_total(self):
        return self.cumulative_frequency[self.symbols[-1]][1]

    def get_low(self, symbol):
        return self.cumulative_frequency[symbol][0]

    def get_high(self, symbol):
        return self.cumulative_frequency[symbol][1]

    def update_table(self, symbol):
        self.freq[symbol] = self.freq.get(symbol, 0) + 1
        self.freq_total += 1
        self.prob = {
            symbol: freq / self.freq_total for symbol, freq in self.freq.items()
        }
        self.cumulative_frequency = self.update_cum_freq()

    def update_cum_freq(self) -> Dict[Union[int, str], Tuple[int, int]]:
        """Create an integer-scaled cummulative distribution function from a probability dist. and scaling factor

        Returns
        -------
        Dict[int, Range]
            A dictionary mapping the token IDs (integers) to their half-open Range of [low, high) corresponding to the scaled probability of the token
        """
        cumulative_frequency = {}
        prev_prob = 0
        scaled_freq = {
            symbol: round(self.scale_factor * prob)
            for symbol, prob in self.prob.items()
        }
        for token, prop in scaled_freq.items():
            # if prop == 0:
            #     """
            #     necessary to handle the situation where due to scaling and rounding of almost zero probabilities, we get tokens
            #     with cumulative_frequency where low == high. This goes against AE's invariant that zero probability symbols are not allowed in the frequency table.
            #     Making sure that the scaled integer interval's width is at least one for all tokens could make the algorithm slightly less efficient
            #     but this efficiency is offset by the algorithm quickly adapting to the true source distribution.
            #     """
            #     prop += 1
            cumulative_frequency[token] = (prev_prob, prev_prob + prop)
            prev_prob += prop
        return cumulative_frequency

    def find_correct_interval(self, target: int):
        symbols = list(self.cumulative_frequency.keys())
        ranges = list(self.cumulative_frequency.values())

        left = 0
        right = len(ranges) - 1
        result = 0
        while left <= right:
            mid = (left + right) // 2
            low, high = ranges[mid]
            if low <= target:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return symbols[result]

    def reset_state(self):
        self.freq_total: int = 0
        self._init_uniform()

    def get_symbol_freq(self, symbol):
        return self.cumulative_frequency[symbol]
